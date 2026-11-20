import test from "node:test";
import assert from "node:assert/strict";
import { once } from "node:events";
import { request as httpRequest, type IncomingMessage } from "node:http";
import { connect } from "node:net";
import { createInterface } from "node:readline";
import { PiMcpServer, type McpCall, type PiMcpServerOptions } from "../src/mcp-server.ts";

const servers: PiMcpServer[] = [];
test.afterEach(() => {
	for (const server of servers.splice(0)) server.dispose();
});

async function startServer(calls: McpCall[] = [], options: Partial<PiMcpServerOptions> = {}): Promise<PiMcpServer> {
	const server = new PiMcpServer({
		tools: () => [{ name: "echo", description: "Echoes", inputSchema: { type: "object" } }],
		call: async (call) => {
			calls.push(call);
			return { content: [{ type: "text", text: "ok" }] };
		},
		...options,
	});
	servers.push(server);
	await server.start();
	return server;
}

const listTools = { jsonrpc: "2.0", id: 1, method: "tools/list", params: {} };

async function post(url: string, token: string): Promise<number> {
	const response = await fetch(url, {
		method: "POST",
		headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
		body: JSON.stringify(listTools),
	});
	await response.arrayBuffer();
	return response.status;
}

test("http MCP answers only a minted, unrevoked nonce of the agent in the path, with that session's token", async () => {
	const server = await startServer();
	const nonce = server.mint("a");
	const entry = server.serverFor("a", "http", nonce);
	assert.ok("url" in entry);
	const token = entry.headers[0]!.value.replace("Bearer ", "");
	const otherNonce = server.mint("a");
	const otherAgent = server.serverFor("b", "http", server.mint("b"));
	assert.ok("url" in otherAgent);
	assert.equal(await post(entry.url, token), 200);
	assert.equal(await post(entry.url, "0".repeat(token.length)), 401, "a wrong token");
	assert.equal(await post(entry.url, token.slice(0, -1)), 401, "a truncated token");
	assert.equal(await post(entry.url.replace(nonce, otherNonce), token), 401, "the token of another nonce of the same agent");
	assert.equal(await post(entry.url.replace("/mcp/a/", "/mcp/b/"), token), 401, "the token on another agent's path");
	assert.equal(await post(otherAgent.url, token), 401, "the token on another agent's session");
	assert.equal(await post(entry.url.replace(nonce, "0".repeat(32)), token), 401, "an unknown nonce");
	assert.equal(await post(entry.url.replace(`/${nonce}`, ""), token), 404, "no nonce");
	server.revoke(nonce);
	assert.equal(await post(entry.url, token), 403, "a revoked nonce");
});

/** Sends headers declaring `declared` bytes (or chunked when undefined), then `sent` bytes, without ending the request; waits for the server to close unless `waitClose` is false. */
async function oversized(url: string, headers: Record<string, string>, sent: number, declared?: number, waitClose = true) {
	const request = httpRequest(url, { method: "POST", headers: declared === undefined ? headers : { ...headers, "content-length": String(declared) } });
	request.on("error", () => {});
	const responded = once(request, "response") as Promise<[IncomingMessage]>;
	const closed = once(request, "close");
	if (sent > 0) request.write("x".repeat(sent));
	else request.flushHeaders();
	const timeout = setTimeout(() => request.destroy(new Error("no answer")), 2000);
	const [response] = await responded;
	response.resume();
	if (waitClose) await closed;
	else request.destroy();
	clearTimeout(timeout);
	return { status: response.statusCode, finished: request.writableFinished };
}

test("http MCP answers 413 to a declared oversized body without reading it, and closes the connection", async () => {
	const server = await startServer([], { maxRequestBytes: 1024 });
	const endpoint = httpEndpoint(server);
	const reply = await oversized(endpoint.url, endpoint.headers, 0, 100 * 1024 * 1024);
	assert.equal(reply.status, 413);
	assert.equal(reply.finished, false, "answered before the client sent the body");
	assert.equal((await endpoint.send(listTools)).status, 200);
});

test("http MCP answers 413 once a chunked body passes the cap, and closes the connection", async () => {
	const server = await startServer([], { maxRequestBytes: 1024 });
	const endpoint = httpEndpoint(server);
	const reply = await oversized(endpoint.url, endpoint.headers, 2048);
	assert.equal(reply.status, 413);
	assert.equal(reply.finished, false, "answered before the client ended the body");
});

test("http MCP refuses an oversized body only after authentication", async () => {
	const server = await startServer([], { maxRequestBytes: 1024 });
	const endpoint = httpEndpoint(server);
	const reply = await oversized(endpoint.url, { ...endpoint.headers, authorization: "Bearer wrong" }, 0, 100 * 1024 * 1024, false);
	assert.equal(reply.status, 401);
});

test("stdio MCP refuses another agent's token and closes an oversized handshake line", async () => {
	const server = await startServer([], { maxRequestBytes: 1024 });
	const nonce = server.mint("a");
	const entry = server.serverFor("a", "stdio", nonce);
	const otherEntry = server.serverFor("b", "stdio", server.mint("b"));
	assert.ok("args" in entry && "args" in otherEntry);
	const envOf = (value: typeof entry) => Object.fromEntries(("env" in value ? value.env : []).map((variable) => [variable.name, variable.value]));
	/** Sends `data` and reports whether the server closed the connection within 300 ms, and what it answered. */
	const handshake = async (data: string) => {
		const socket = connect(entry.args[1]!);
		await once(socket, "connect");
		socket.on("error", () => {});
		const replies: string[] = [];
		createInterface({ input: socket }).on("line", (reply) => replies.push(reply)).on("error", () => {});
		const closed = once(socket, "close").then(() => true);
		socket.write(data);
		const wasClosed = await Promise.race([closed, new Promise<boolean>((resolve) => setTimeout(() => resolve(false), 300))]);
		socket.destroy();
		return { closed: wasClosed, replies };
	};
	const withCall = (hello: object) => `${JSON.stringify(hello)}\n${JSON.stringify(listTools)}\n`;
	const foreign = await handshake(withCall({ token: envOf(otherEntry).PI_MCP_TOKEN, agent: "a", session: nonce }));
	assert.equal(foreign.closed, true, "another agent's token");
	assert.deepEqual(foreign.replies, []);
	const huge = await handshake(`{"token":"${"x".repeat(8192)}`);
	assert.equal(huge.closed, true, "an unterminated handshake line over the cap");
	const good = await handshake(withCall({ token: envOf(entry).PI_MCP_TOKEN, agent: "a", session: nonce }));
	assert.equal(good.replies.length, 1, "the session's own token");
	assert.equal(good.closed, false);
});

test("http MCP refuses a request whose nonce was revoked while its body uploaded", async () => {
	const server = await startServer();
	const nonce = server.mint("a");
	const entry = server.serverFor("a", "http", nonce);
	assert.ok("url" in entry);
	const body = JSON.stringify(listTools);
	const request = httpRequest(entry.url, {
		method: "POST",
		headers: { ...Object.fromEntries(entry.headers.map((header) => [header.name, header.value])), "content-type": "application/json" },
	});
	const responded = once(request, "response") as Promise<[IncomingMessage]>;
	request.write(body.slice(0, 5));
	await new Promise((resolve) => setTimeout(resolve, 50));
	server.revoke(nonce);
	request.end(body.slice(5));
	const [response] = await responded;
	response.resume();
	assert.equal(response.statusCode, 403);
});

/** An HTTP endpoint for a freshly minted nonce of agent "a", with helpers that never hang a failing test. */
function httpEndpoint(server: PiMcpServer) {
	const nonce = server.mint("a");
	const entry = server.serverFor("a", "http", nonce);
	assert.ok("url" in entry);
	const headers: Record<string, string> = {
		...Object.fromEntries(entry.headers.map((header) => [header.name, header.value])),
		"content-type": "application/json",
	};
	const send = async (body: unknown, session?: string, method = "POST", url = entry.url) => {
		const response = await fetch(url, {
			method,
			headers: session ? { ...headers, "mcp-session-id": session } : headers,
			...(method === "POST" ? { body: JSON.stringify(body) } : {}),
			signal: AbortSignal.timeout(2000),
		});
		const text = await response.text();
		return { status: response.status, session: response.headers.get("mcp-session-id"), json: text ? (JSON.parse(text) as Record<string, any>) : null };
	};
	return { nonce, url: entry.url, headers, send };
}

const initialize = { jsonrpc: "2.0", id: 1, method: "initialize", params: { protocolVersion: "2025-06-18" } };
const callEcho = (id: number) => ({ jsonrpc: "2.0", id, method: "tools/call", params: { name: "echo", arguments: {} } });

/** Records unhandled rejections while `body` runs, so a crash of Pi's process shows up as a test failure. */
async function withoutUnhandledRejections(body: () => Promise<void>): Promise<void> {
	const rejections: unknown[] = [];
	const listener = (reason: unknown) => rejections.push(reason);
	process.on("unhandledRejection", listener);
	try {
		await body();
		await new Promise((resolve) => setTimeout(resolve, 50));
	} finally {
		process.off("unhandledRejection", listener);
	}
	assert.deepEqual(rejections, []);
}

/** A call handler that stays pending until its signal aborts; `aborted` resolves with the aborted call. */
function pendingCalls() {
	let started!: (call: McpCall) => void;
	const running = new Promise<McpCall>((resolve) => (started = resolve));
	const call = (mcpCall: McpCall) =>
		new Promise<never>((_resolve, reject) => {
			started(mcpCall);
			mcpCall.signal.addEventListener("abort", () => reject(new Error("cancelled")), { once: true });
		});
	return { call, running };
}

test("http MCP answers 400 to a malformed agent path and keeps serving", async () => {
	const server = await startServer();
	const endpoint = httpEndpoint(server);
	await withoutUnhandledRejections(async () => {
		const malformed = await endpoint.send(listTools, undefined, "POST", endpoint.url.replace("/mcp/a/", "/mcp/%E0/"));
		assert.equal(malformed.status, 400);
	});
	assert.equal((await endpoint.send(listTools)).status, 200);
});

test("http MCP survives a client that aborts mid-upload", async () => {
	const server = await startServer();
	const endpoint = httpEndpoint(server);
	await withoutUnhandledRejections(async () => {
		const request = httpRequest(endpoint.url, { method: "POST", headers: { ...endpoint.headers, "content-length": "1000" } });
		request.on("error", () => {});
		request.write("{\"jsonrpc\":");
		await new Promise((resolve) => setTimeout(resolve, 50));
		request.destroy();
	});
	assert.equal((await endpoint.send(listTools)).status, 200);
});

test("http and stdio MCP answer a throwing handler with a JSON-RPC error", async () => {
	const server = await startServer([], {
		tools: () => {
			throw new Error("boom");
		},
	});
	const endpoint = httpEndpoint(server);
	await withoutUnhandledRejections(async () => {
		const reply = await endpoint.send(listTools);
		assert.equal(reply.status, 200);
		assert.equal(reply.json?.error?.code, -32603);
		const stdio = await relay(server, "a", endpoint.nonce, endpoint.nonce);
		assert.equal((stdio.replies[0] as Record<string, any> | undefined)?.error?.code, -32603);
		stdio.socket.destroy();
	});
});

test("http MCP cancels a call from a separate POST of the same MCP session", async () => {
	const pending = pendingCalls();
	const server = await startServer([], { call: pending.call });
	const endpoint = httpEndpoint(server);
	const init = await endpoint.send(initialize);
	assert.ok(init.session, "initialize issues an Mcp-Session-Id");
	const reply = endpoint.send(callEcho(7), init.session!);
	const call = await pending.running;
	assert.equal((await endpoint.send({ jsonrpc: "2.0", method: "notifications/cancelled", params: { requestId: 7 } }, init.session!)).status, 202);
	assert.ok(call.signal.aborted);
	assert.equal((await reply).json?.result?.isError, true);
});

test("http MCP aborts a call in flight when its nonce is revoked", async () => {
	const pending = pendingCalls();
	const server = await startServer([], { call: pending.call });
	const endpoint = httpEndpoint(server);
	const init = await endpoint.send(initialize);
	const reply = endpoint.send(callEcho(7), init.session!);
	const call = await pending.running;
	server.revoke(endpoint.nonce);
	assert.ok(call.signal.aborted);
	assert.equal((await reply).json?.result?.isError, true);
	assert.equal((await endpoint.send(listTools, init.session!)).status, 403);
});

test("http MCP DELETE aborts the MCP session's calls and ends the session", async () => {
	const pending = pendingCalls();
	const server = await startServer([], { call: pending.call });
	const endpoint = httpEndpoint(server);
	const init = await endpoint.send(initialize);
	const reply = endpoint.send(callEcho(7), init.session!);
	const call = await pending.running;
	assert.equal((await endpoint.send(null, init.session!, "DELETE")).status, 200);
	assert.ok(call.signal.aborted);
	assert.equal((await reply).json?.result?.isError, true);
	assert.equal((await endpoint.send(listTools, init.session!)).status, 404, "an ended MCP session");
	assert.equal((await endpoint.send(listTools)).status, 200, "a request without a session");
});

test("http MCP answers 404 to a request whose MCP session was deleted while its body uploaded", async () => {
	const server = await startServer();
	const endpoint = httpEndpoint(server);
	const init = await endpoint.send(initialize);
	const body = JSON.stringify(listTools);
	const request = httpRequest(endpoint.url, { method: "POST", headers: { ...endpoint.headers, "mcp-session-id": init.session! } });
	const responded = once(request, "response") as Promise<[IncomingMessage]>;
	request.write(body.slice(0, 5));
	await new Promise((resolve) => setTimeout(resolve, 50));
	assert.equal((await endpoint.send(null, init.session!, "DELETE")).status, 200);
	request.end(body.slice(5));
	const [response] = await responded;
	response.resume();
	assert.equal(response.statusCode, 404);
});

/** Opens a raw stdio-relay connection and sends its handshake; resolves once the server answered or closed. */
async function relay(server: PiMcpServer, agentId: string, nonce: string, session: string) {
	const entry = server.serverFor(agentId, "stdio", nonce);
	assert.ok("args" in entry);
	const env = Object.fromEntries(entry.env.map((variable) => [variable.name, variable.value]));
	assert.equal(env.PI_MCP_SESSION, nonce);
	const socket = connect(entry.args[1]!);
	await once(socket, "connect");
	const closed = once(socket, "close");
	const lines = createInterface({ input: socket });
	lines.on("error", () => {});
	const replies: unknown[] = [];
	lines.on("line", (line) => replies.push(JSON.parse(line)));
	socket.write(`${JSON.stringify({ token: env.PI_MCP_TOKEN, agent: env.PI_MCP_AGENT, session })}\n`);
	socket.write(`${JSON.stringify(listTools)}\n`);
	const answered = new Promise<void>((resolve) => lines.once("line", () => resolve()));
	await Promise.race([answered, closed]);
	return { socket, closed, replies };
}

test("stdio MCP refuses a wrong nonce and closes a connection once its nonce is revoked", async () => {
	const server = await startServer();
	const nonce = server.mint("a");
	const wrong = await relay(server, "a", nonce, "f".repeat(32));
	await wrong.closed;
	assert.deepEqual(wrong.replies, []);

	const good = await relay(server, "a", nonce, nonce);
	assert.equal(good.replies.length, 1);
	server.revoke(nonce);
	await good.closed;
	assert.ok(good.socket.destroyed);
});
