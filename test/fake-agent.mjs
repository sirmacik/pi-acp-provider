/**
 * Fake ACP (v1) agent for tests.
 *
 * Behavior is driven by markers in the prompt text and FAKE_* env vars:
 *   - "PERMISSION"  asks session/request_permission (string request id) first
 *   - "SLOW"        delays each streamed update (~150ms)
 *   - "CRASH"       exits (code 3) mid-stream after the tool update
 *   - "AUTH"        fails the prompt with the ACP auth_required error (-32000)
 *   - "RPCERROR"    fails the prompt with an internal error
 *   - "MAXTOKENS"   finishes with stopReason "max_tokens"
 *   - "REFUSAL"     finishes with stopReason "refusal"
 *   - "ONESHOT"     first reports a tool call that is already completed
 *   - "DIFF"        first reports an edit tool call with a diff
 *   - FAKE_SPLIT=1  writes stdout in small split chunks (line-buffer test)
 *   - FAKE_MODELS=config  session/new returns configOptions (model + thought_level)
 *     FAKE_MODELS=legacy  session/new returns the legacy `models` field
 *                   Replies end with "[model=<id> thought=<level>]".
 *     FAKE_DEFAULT_MODEL=<id>  the model new sessions start on (default "fast")
 *   - "/context"    (whole prompt) answers like an agent slash command
 *   - FAKE_PROTOCOL=n answers initialize with protocol version n
 *   - FAKE_AUTH=1   session/new fails with auth_required until `authenticate`
 *                   is called with methodId "fake-login"
 *   - "ECHO"        replies "ECHO[<session id>]<<<full prompt text>>>" and ends the turn
 *   - "SESSIONINFO" replies with JSON of the session's session/new params
 *   - "MCPCALL:<tool>:<json args>"  reports a tool call titled "mcp__pi__<tool>",
 *                   calls <tool> on the session's MCP server (stdio or http),
 *                   then replies "RESULT<<<text>>>" (plus "STEERED<<<text>>>"
 *                   when a _session/steering message arrived meanwhile)
 *   - FAKE_MCP_TITLE=<t> titles that tool call <t> instead
 *   - FAKE_MCP_HTTP=1    advertises HTTP MCP support
 *   - FAKE_SYSPROMPT=1   advertises sessionCapabilities.systemPrompt
 *   - FAKE_STEERING=1    advertises and accepts `_session/steering`
 */

import { spawn } from "node:child_process";
import { createInterface } from "node:readline";

const SPLIT = process.env.FAKE_SPLIT === "1";
let authenticated = process.env.FAKE_AUTH !== "1";

function emit(message) {
	const line = JSON.stringify(message) + "\n";
	if (!SPLIT) {
		process.stdout.write(line);
		return;
	}
	for (let index = 0; index < line.length; index += 7) {
		process.stdout.write(line.slice(index, index + 7));
	}
}

const reply = (id, result) => emit({ jsonrpc: "2.0", id, result });
const replyError = (id, code, message) => emit({ jsonrpc: "2.0", id, error: { code, message } });
const update = (sessionId, sessionUpdate) =>
	emit({ jsonrpc: "2.0", method: "session/update", params: { sessionId, update: sessionUpdate } });
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const MODELS = process.env.FAKE_MODELS;
const settings = { model: process.env.FAKE_DEFAULT_MODEL ?? "fast", thought: "medium" };
const configOptions = () => [
	{
		id: "model",
		name: "Model",
		category: "model",
		type: "select",
		currentValue: settings.model,
		options: [
			{ group: "main", name: "Main", options: [{ value: "fast", name: "Fast" }] },
			{ value: "smart", name: "Smart", description: "Slow but clever" },
		],
	},
	{
		id: "effort",
		name: "Reasoning",
		category: "thought_level",
		type: "select",
		currentValue: settings.thought,
		options: ["none", "low", "medium", "high"].map((value) => ({ value, name: value })),
	},
];
const sessionExtras = () =>
	MODELS === "config"
		? { configOptions: configOptions() }
		: MODELS === "legacy"
			? { models: { currentModelId: settings.model, availableModels: [{ modelId: "fast", name: "Fast" }, { modelId: "smart", name: "Smart" }] } }
			: {};

let sessionCounter = 0;
let promptCounter = 0;
let serverRequestId = 0;
const serverRequestWaiters = new Map();
let activePrompt = null;
const sessionParams = new Map();
const steered = [];

/** A tiny MCP client: initialize, tools/list, tools/call against one ACP McpServer entry. */
async function mcpCall(server, name, args) {
	const init = { protocolVersion: "2025-06-18", capabilities: {}, clientInfo: { name: "fake-agent", version: "0" } };
	if (server.type === "http") {
		const headers = { "content-type": "application/json", accept: "application/json, text/event-stream" };
		for (const header of server.headers ?? []) headers[header.name] = header.value;
		const post = async (body) => {
			const response = await fetch(server.url, { method: "POST", headers, body: JSON.stringify(body) });
			return response.status === 202 ? null : response.json();
		};
		await post({ jsonrpc: "2.0", id: 1, method: "initialize", params: init });
		await post({ jsonrpc: "2.0", method: "notifications/initialized" });
		const list = await post({ jsonrpc: "2.0", id: 2, method: "tools/list" });
		const call = await post({ jsonrpc: "2.0", id: 3, method: "tools/call", params: { name, arguments: args } });
		return { tools: list.result.tools, result: call.result };
	}
	const env = { ...process.env };
	for (const entry of server.env ?? []) env[entry.name] = entry.value;
	const child = spawn(server.command, server.args, { env, stdio: ["pipe", "pipe", "inherit"] });
	const waiters = new Map();
	createInterface({ input: child.stdout }).on("line", (line) => {
		const message = JSON.parse(line);
		waiters.get(message.id)?.(message);
	});
	const request = (id, method, params) =>
		new Promise((resolve) => {
			waiters.set(id, resolve);
			child.stdin.write(JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n");
		});
	try {
		await request(1, "initialize", init);
		child.stdin.write(JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" }) + "\n");
		const list = await request(2, "tools/list", {});
		const call = await request(3, "tools/call", { name, arguments: args });
		return { tools: list.result.tools, result: call.result };
	} finally {
		child.kill();
	}
}

function serverRequest(method, params) {
	return new Promise((resolve) => {
		const id = `srv-${++serverRequestId}`;
		serverRequestWaiters.set(id, resolve);
		emit({ jsonrpc: "2.0", id, method, params });
	});
}

async function handlePrompt(message) {
	const { sessionId } = message.params;
	const text = (message.params.prompt ?? []).map((block) => block?.text ?? "").join("");
	const turn = {
		finished: false,
		finish(stopReason) {
			if (this.finished) return;
			this.finished = true;
			reply(message.id, { stopReason });
		},
	};
	activePrompt = turn;
	const stepDelay = text.includes("SLOW") ? 150 : 5;

	if (text === "/context") {
		update(sessionId, { sessionUpdate: "agent_message_chunk", content: { type: "text", text: "context: 42%" } });
		return turn.finish("end_turn");
	}
	if (text.includes("ECHO")) {
		update(sessionId, { sessionUpdate: "agent_message_chunk", content: { type: "text", text: `ECHO[${sessionId}]<<<${text}>>>` } });
		promptCounter++;
		return turn.finish("end_turn");
	}
	if (text.includes("SESSIONINFO")) {
		update(sessionId, { sessionUpdate: "agent_message_chunk", content: { type: "text", text: JSON.stringify(sessionParams.get(sessionId) ?? null) } });
		return turn.finish("end_turn");
	}
	const mcpMatch = /MCPCALL:([\w-]+):(\{.*?\})(?:$|\s)/.exec(text);
	if (mcpMatch) {
		const [, tool, json] = mcpMatch;
		const args = JSON.parse(json);
		update(sessionId, { sessionUpdate: "tool_call", toolCallId: "m1", title: process.env.FAKE_MCP_TITLE ?? `mcp__pi__${tool}`, status: "pending", rawInput: args });
		await sleep(20);
		const server = (sessionParams.get(sessionId)?.mcpServers ?? []).find((entry) => entry.name === "pi");
		if (!server) {
			update(sessionId, { sessionUpdate: "agent_message_chunk", content: { type: "text", text: "NO-MCP-SERVER" } });
			return turn.finish("end_turn");
		}
		const { tools, result } = await mcpCall(server, tool, args);
		update(sessionId, { sessionUpdate: "tool_call_update", toolCallId: "m1", status: result.isError ? "failed" : "completed" });
		const resultText = (result.content ?? []).map((block) => (block.type === "text" ? block.text : `[${block.type}]`)).join("|");
		const extra = steered.length > 0 ? ` STEERED<<<${steered.splice(0).join("|")}>>>` : "";
		update(sessionId, {
			sessionUpdate: "agent_message_chunk",
			content: { type: "text", text: `TOOLS<<<${tools.map((entry) => entry.name).join(",")}>>> RESULT<<<${resultText}>>>${extra}` },
		});
		promptCounter++;
		return turn.finish("end_turn");
	}
	if (text.includes("AUTH")) return replyError(message.id, -32000, "Authentication required");
	if (text.includes("RPCERROR")) return replyError(message.id, -32603, "fake rpc failure");

	let approved = null;
	if (text.includes("PERMISSION")) {
		const response = await serverRequest("session/request_permission", {
			sessionId,
			toolCall: { toolCallId: "t0", title: "rm -rf build" },
			options: [
				{ optionId: "allow", name: "Allow", kind: "allow_once" },
				{ optionId: "allow-all", name: "Always allow", kind: "allow_always" },
				{ optionId: "deny", name: "Deny", kind: "reject_once" },
			],
		});
		const outcome = response.result?.outcome ?? {};
		approved = outcome.outcome === "selected" ? outcome.optionId : outcome.outcome ?? "error";
		if (turn.finished) return;
	}

	const steps = [
		{ sessionUpdate: "agent_thought_chunk", content: { type: "text", text: "Considering the request... " } },
		{ sessionUpdate: "tool_call", toolCallId: "t1", title: "grep for secrets", kind: "search", status: "pending" },
		{ sessionUpdate: "tool_call_update", toolCallId: "t1", status: "in_progress" },
		{
			sessionUpdate: "tool_call_update",
			toolCallId: "t1",
			status: "completed",
			content: [{ type: "content", content: { type: "text", text: "3 matches\nsecond line" } }],
		},
		{ sessionUpdate: "plan", entries: [] },
		{ sessionUpdate: "agent_message_chunk", content: { type: "text", text: "Hello " } },
		{
			sessionUpdate: "agent_message_chunk",
			content: { type: "text", text: `world #${promptCounter}${approved !== null ? ` [approved=${approved}]` : ""}${MODELS ? ` [model=${settings.model} thought=${settings.thought}]` : ""}` },
		},
	];
	if (text.includes("ONESHOT")) {
		steps.unshift({
			sessionUpdate: "tool_call",
			toolCallId: "t0",
			title: "read notes",
			status: "completed",
			content: [{ type: "content", content: { type: "text", text: "2 notes" } }],
		});
	}
	if (text.includes("DIFF")) {
		steps.unshift(
			{ sessionUpdate: "tool_call", toolCallId: "e1", title: "Edit notes.txt", kind: "edit", status: "in_progress", locations: [{ path: "/tmp/notes.txt", line: 2 }] },
			{
				sessionUpdate: "tool_call_update",
				toolCallId: "e1",
				status: "completed",
				content: [{ type: "diff", path: "/tmp/notes.txt", oldText: "one\ntwo\n", newText: "one\nTWO\n" }],
			},
		);
	}
	for (const step of steps) {
		if (turn.finished) return;
		update(sessionId, step);
		if (step.status === "completed" && text.includes("CRASH")) process.exit(3);
		await sleep(stepDelay);
	}
	promptCounter++;
	turn.finish(text.includes("MAXTOKENS") ? "max_tokens" : text.includes("REFUSAL") ? "refusal" : "end_turn");
}

createInterface({ input: process.stdin }).on("line", (line) => {
	let message;
	try {
		message = JSON.parse(line);
	} catch {
		return;
	}
	if (message.method === undefined) {
		const waiter = serverRequestWaiters.get(message.id);
		serverRequestWaiters.delete(message.id);
		waiter?.(message);
		return;
	}
	switch (message.method) {
		case "initialize":
			reply(message.id, {
				protocolVersion: Number(process.env.FAKE_PROTOCOL ?? 1),
				agentCapabilities: {
					loadSession: false,
					promptCapabilities: { image: true },
					...(process.env.FAKE_MCP_HTTP === "1" ? { mcpCapabilities: { http: true } } : {}),
					...(process.env.FAKE_SYSPROMPT === "1" ? { sessionCapabilities: { systemPrompt: {} } } : {}),
				},
				...(process.env.FAKE_STEERING === "1" ? { _meta: { steering: { supported: true } } } : {}),
				agentInfo: { name: `fake-agent-${process.pid}`, version: "0.0.1" },
				authMethods: [{ id: "fake-login", name: "Fake login", description: "Pretend to sign in" }],
			});
			break;
		case "authenticate":
			if (message.params?.methodId !== "fake-login") {
				replyError(message.id, -32602, "unknown auth method");
				break;
			}
			authenticated = true;
			reply(message.id, {});
			break;
		case "session/new":
			if (!authenticated) replyError(message.id, -32000, "Authentication required");
			else {
				const sessionId = `sess-${++sessionCounter}`;
				sessionParams.set(sessionId, message.params);
				reply(message.id, { sessionId, ...sessionExtras() });
				update(sessionId, {
					sessionUpdate: "available_commands_update",
					availableCommands: [
						{ name: "context", description: "Show context usage" },
						{ name: "session", description: "Session tools", input: { hint: "info | rename" } },
					],
				});
			}
			break;
		case "session/set_config_option": {
			const { configId, value } = message.params;
			if (MODELS !== "config" || (configId !== "model" && configId !== "effort")) {
				replyError(message.id, -32602, "unknown config option");
				break;
			}
			settings[configId === "model" ? "model" : "thought"] = value;
			reply(message.id, { configOptions: configOptions() });
			break;
		}
		case "session/set_model":
			if (MODELS !== "legacy") replyError(message.id, -32601, "Method not found");
			else {
				settings.model = message.params.modelId;
				reply(message.id, {});
			}
			break;
		case "session/prompt":
			void handlePrompt(message);
			break;
		case "_session/steering": {
			if (process.env.FAKE_STEERING !== "1") {
				replyError(message.id, -32601, "Method not found");
				break;
			}
			const steerText = (message.params.prompt ?? []).map((block) => block?.text ?? "").join("");
			if (activePrompt && !activePrompt.finished) {
				steered.push(steerText);
				reply(message.id, { outcome: "steered" });
			} else reply(message.id, { outcome: "promptRequired", reason: "noRunningTurn" });
			break;
		}
		case "session/cancel":
			activePrompt?.finish("cancelled");
			break;
		default:
			if (message.id !== undefined) replyError(message.id, -32601, `Method not found: ${message.method}`);
	}
});
