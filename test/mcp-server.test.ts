import test from "node:test";
import assert from "node:assert/strict";
import { once } from "node:events";
import { request as httpRequest, type IncomingMessage } from "node:http";
import { connect } from "node:net";
import { createInterface } from "node:readline";
import { PiMcpServer, type McpCall } from "../src/mcp-server.ts";

const servers: PiMcpServer[] = [];
test.afterEach(() => {
	for (const server of servers.splice(0)) server.dispose();
});

async function startServer(calls: McpCall[] = []): Promise<PiMcpServer> {
	const server = new PiMcpServer({
		tools: () => [{ name: "echo", description: "Echoes", inputSchema: { type: "object" } }],
		call: async (call) => {
			calls.push(call);
			return { content: [{ type: "text", text: "ok" }] };
		},
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

test("http MCP answers only a minted, unrevoked nonce of the agent in the path", async () => {
	const server = await startServer();
	const nonce = server.mint("a");
	const entry = server.serverFor("a", "http", nonce);
	assert.ok("url" in entry);
	const token = entry.headers[0]!.value.replace("Bearer ", "");
	assert.equal(await post(entry.url, token), 200);
	assert.equal(await post(entry.url.replace(nonce, "0".repeat(32)), token), 403, "an unknown nonce");
	assert.equal(await post(entry.url.replace("/mcp/a/", "/mcp/b/"), token), 403, "another agent's nonce");
	assert.equal(await post(entry.url.replace(`/${nonce}`, ""), token), 404, "no nonce");
	server.revoke(nonce);
	assert.equal(await post(entry.url, token), 403, "a revoked nonce");
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
