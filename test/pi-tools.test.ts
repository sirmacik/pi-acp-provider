import test from "node:test";
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { normalizeContext } from "@earendil-works/pi-ai/utils/transcript";
import type { AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream, Message, Model, Api, Tool } from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { createAcpProvider, piToolRef, type AcpProviderHandle, type AcpProviderOptions } from "../src/provider.ts";
import type { AcpAgentConfig } from "../src/config.ts";
import { toToolResult } from "../src/native-tools.ts";
import { flattenRootCombinators, toMcpTool } from "../src/mcp-server.ts";

const fakePath = fileURLToPath(new URL("./fake-agent.mjs", import.meta.url));

function agent(overrides: Partial<AcpAgentConfig> = {}): AcpAgentConfig {
	return {
		id: "fake",
		name: "Fake agent",
		command: process.execPath,
		args: [fakePath],
		env: {},
		authMethod: undefined,
		piTools: "both",
		mcpTransport: "stdio",
		systemPrompt: "prompt",
		sessionMeta: undefined,
		ownToolsOffMeta: undefined,
		...overrides,
	};
}

const handles: AcpProviderHandle[] = [];
test.afterEach(() => {
	for (const handle of handles.splice(0)) handle.dispose();
});

function makeProvider(agents: AcpAgentConfig[], extra: Omit<AcpProviderOptions, "config"> = {}): AcpProviderHandle {
	const handle = createAcpProvider({
		config: { agents, autoApproveTools: true, handshakeTimeoutMs: 8000, debug: false, nativeTools: true },
		...extra,
	});
	handles.push(handle);
	return handle;
}

const echoTool: Tool = {
	name: "echo",
	description: "Echoes text",
	parameters: Type.Object({ text: Type.String() }),
};

let clock = Date.now();
const user = (text: string): Message => ({ role: "user", content: [{ type: "text", text }], timestamp: ++clock });

async function collect(stream: AssistantMessageEventStream): Promise<AssistantMessageEvent[]> {
	const events: AssistantMessageEvent[] = [];
	for await (const event of stream) events.push(event);
	return events;
}

function model(handle: AcpProviderHandle): Model<Api> {
	const first = handle.provider.getModels()[0];
	assert.ok(first);
	return first;
}

const textOf = (message: AssistantMessage) => message.content.flatMap((block) => (block.type === "text" ? [block.text] : [])).join("");

/** One provider call; returns the final event, which must be `done`. */
async function step(handle: AcpProviderHandle, messages: Message[], systemPrompt?: string, options: Record<string, unknown> = {}) {
	const context = normalizeContext({ ...(systemPrompt ? { systemPrompt } : {}), tools: [echoTool], messages });
	const last = (await collect(handle.provider.streamSimple(model(handle), context, options))).at(-1);
	assert.ok(last?.type === "done", `expected done, got ${JSON.stringify(last)}`);
	return last.message;
}

/** Answers every tool call of `message` the way Pi would: echo runs, acp_tool waits on the agent. */
async function answerTools(handle: AcpProviderHandle, message: AssistantMessage): Promise<Message[]> {
	const results: Message[] = [];
	for (const block of message.content) {
		if (block.type !== "toolCall") continue;
		const content =
			block.name === "echo"
				? [{ type: "text" as const, text: `echoed ${(block.arguments as { text: string }).text}` }]
				: toToolResult(await handle.awaitTool(block.id, undefined, () => {})).content;
		results.push({ role: "toolResult", toolCallId: block.id, toolName: block.name, content, isError: false, timestamp: ++clock } as Message);
	}
	return results;
}

for (const transport of ["stdio", "http"] as const) {
	test(`a Pi tool called over ${transport} MCP runs as a Pi tool call and its result reaches the agent`, async () => {
		const handle = makeProvider([agent({ mcpTransport: transport, env: transport === "http" ? { FAKE_MCP_HTTP: "1" } : {} })]);
		const messages: Message[] = [user('MCPCALL:echo:{"text":"hi"}')];
		const first = await step(handle, messages);
		assert.equal(first.stopReason, "toolUse");
		const calls = first.content.filter((block) => block.type === "toolCall");
		assert.equal(calls.length, 1, "the agent's own report of the MCP call is not mirrored as acp_tool");
		assert.equal(calls[0]?.type === "toolCall" && calls[0].name, "echo");
		assert.deepEqual(calls[0]?.type === "toolCall" && calls[0].arguments, { text: "hi" });
		messages.push(first, ...(await answerTools(handle, first)));
		const last = await step(handle, messages);
		assert.equal(last.stopReason, "stop");
		assert.match(textOf(last), /TOOLS<<<echo>>> RESULT<<<echoed hi>>>/);
	});
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

for (const [knob, label, call] of [
	// The cancellation must have reached the provider, which rejects the call; not a call left pending.
	["FAKE_MCP_CANCEL", "cancelled its MCP call", /CALL<<<The tool call was cancelled\.>>>/],
	["FAKE_MCP_GIVEUP", "ended the turn without waiting", /CALL<<<error:gave up>>>/],
] as const) {
	test(`a Pi tool result after the agent ${label} ends the turn with the agent's text`, async () => {
		const handle = makeProvider([agent({ env: { [knob]: "1" } })]);
		const messages: Message[] = [user('MCPCALL:echo:{"text":"hi"}')];
		const first = await step(handle, messages);
		assert.equal(first.stopReason, "toolUse");
		// Pi is still running the tool when the agent stops waiting for it.
		await sleep(300);
		messages.push(first, ...(await answerTools(handle, first)));
		const last = await step(handle, messages);
		assert.equal(last.stopReason, "stop");
		assert.match(textOf(last), /AGENT-WENT-ON/);
		assert.match(textOf(last), call);
	});
}

/** Streams `messages` and aborts on the first thinking delta; returns the aborted message. */
async function abortedStep(handle: AcpProviderHandle, messages: Message[]): Promise<AssistantMessage> {
	const controller = new AbortController();
	const context = normalizeContext({ tools: [echoTool], messages });
	let last: AssistantMessageEvent | undefined;
	for await (const event of handle.provider.streamSimple(model(handle), context, { signal: controller.signal })) {
		last = event;
		if (event.type === "thinking_delta") controller.abort();
	}
	assert.ok(last?.type === "error" && last.reason === "aborted", `expected aborted, got ${JSON.stringify(last)}`);
	return last.error;
}

test("a cancelled prompt's late updates, permission requests and MCP calls stay out of the next turn", async () => {
	const asked: string[] = [];
	const handle = makeProvider([agent({ env: { FAKE_CANCEL_LAG: "200" } })], {
		permissionPolicy: async (request) => {
			asked.push(request.title);
			return { outcome: "selected", optionId: "allow" };
		},
	});
	const messages: Message[] = [user("SLOW work")];
	messages.push(await abortedStep(handle, messages), user("ECHO next"));
	const next = await step(handle, messages);
	assert.equal(next.stopReason, "stop");
	const text = textOf(next);
	assert.doesNotMatch(text, /LATE-CHUNK/);
	assert.match(text, /LAG<<<permission=cancelled;mcp=rejected>>>/);
	assert.deepEqual(asked, []);
	assert.deepEqual((await handle.commands("fake")).map((command) => command.name), ["context", "session"]);
});

for (const transport of ["stdio", "http"] as const) {
	test(`a session whose cancel never drains is replaced and its later ${transport} MCP, permission and update traffic is refused`, async () => {
		const asked: string[] = [];
		const handle = makeProvider(
			[agent({ mcpTransport: transport, env: { FAKE_IGNORE_CANCEL: "1", ...(transport === "http" ? { FAKE_MCP_HTTP: "1" } : {}) } })],
			{
				cancelGraceMs: 500,
				permissionPolicy: async (request) => {
					asked.push(request.title);
					return { outcome: "selected", optionId: "allow" };
				},
			},
		);
		const messages: Message[] = [user("SLOW work")];
		messages.push(await abortedStep(handle, messages), user("ECHO next"));
		const next = await step(handle, messages);
		assert.equal(next.stopReason, "stop");
		const text = textOf(next);
		assert.match(text, /^ECHO\[sess-2\]/, "the next prompt runs on a new session");
		assert.match(text, /<conversation-history>[\s\S]*SLOW work[\s\S]*<\/conversation-history>/, "the new session is re-seeded");
		assert.doesNotMatch(text, /STALE-CHUNK/);
		assert.match(text, /STALE<<<permission=cancelled;mcp=rejected>>>/);
		assert.deepEqual(asked, []);
	});
}

test("an unrecognized report of a Pi tool call that Pi mirrors does not block the call", async () => {
	const handle = makeProvider([agent({ env: { FAKE_MCP_TITLE: "Run something" } })], { nativeTools: () => true });
	const messages: Message[] = [user('MCPCALL:echo:{"text":"hi"}')];
	const names: string[] = [];
	for (let index = 0; index < 5; index++) {
		const message = await step(handle, messages);
		names.push(...message.content.flatMap((block) => (block.type === "toolCall" ? [block.name] : [])));
		messages.push(message);
		if (message.stopReason !== "toolUse") {
			assert.match(textOf(message), /RESULT<<<echoed hi>>>/);
			assert.deepEqual(names, ["acp_tool", "echo"]);
			return;
		}
		messages.push(...(await answerTools(handle, message)));
	}
	assert.fail("the turn did not end");
});

test("pi tools off: no MCP server is offered", async () => {
	const handle = makeProvider([agent({ piTools: "off" })]);
	const message = await step(handle, [user("SESSIONINFO")]);
	assert.deepEqual(JSON.parse(textOf(message)).mcpServers, []);
});

test("pi mode merges the agent's tool switch into the session meta; both mode does not", async () => {
	const off = { vendor: { tools: [] } };
	const piOnly = makeProvider([agent({ piTools: "pi", ownToolsOffMeta: off, sessionMeta: { vendor: { allowed: ["mcp__pi"] } } })]);
	const piInfo = JSON.parse(textOf(await step(piOnly, [user("SESSIONINFO")])));
	assert.deepEqual(piInfo._meta, { vendor: { allowed: ["mcp__pi"], tools: [] } });
	const both = makeProvider([agent({ id: "both", ownToolsOffMeta: off })]);
	const bothInfo = JSON.parse(textOf(await step(both, [user("SESSIONINFO")])));
	assert.equal(bothInfo._meta, undefined);
});

test("http transport falls back to stdio when the agent does not advertise it", async () => {
	const handle = makeProvider([agent({ mcpTransport: "http" })]);
	const info = JSON.parse(textOf(await step(handle, [user("SESSIONINFO")])));
	assert.equal(info.mcpServers[0].type, undefined);
	assert.equal(info.mcpServers[0].name, "pi");
});

test("prompt mode: the system prompt goes in front of the first prompt only", async () => {
	const handle = makeProvider([agent()]);
	const messages: Message[] = [user("ECHO one")];
	const first = await step(handle, messages, "SYSTEM-RULES");
	assert.match(textOf(first), /<system-instructions>[\s\S]*SYSTEM-RULES[\s\S]*<\/system-instructions>ECHO one/);
	messages.push(first, user("ECHO two"));
	const second = await step(handle, messages, "SYSTEM-RULES");
	assert.doesNotMatch(textOf(second), /system-instructions/);
	messages.push(second, user("ECHO three"));
	const third = await step(handle, messages, "NEW-RULES");
	assert.match(textOf(third), /updated its instructions[\s\S]*NEW-RULES/);
});

test("session mode: the system prompt goes on session/new when the agent supports it", async () => {
	const handle = makeProvider([agent({ systemPrompt: "session", env: { FAKE_SYSPROMPT: "1" } })]);
	const messages: Message[] = [user("SESSIONINFO")];
	const info = JSON.parse(textOf(await step(handle, messages, "SYSTEM-RULES")));
	assert.deepEqual(info.systemPrompt, { mode: "append", content: [{ type: "text", text: "SYSTEM-RULES" }] });
	const echo = await step(handle, [...messages, user("ECHO x")], "SYSTEM-RULES");
	assert.doesNotMatch(textOf(echo), /system-instructions/);
});

test("session mode falls back to the prompt when the agent does not advertise it", async () => {
	const handle = makeProvider([agent({ systemPrompt: "session" })]);
	assert.match(textOf(await step(handle, [user("ECHO x")], "SYSTEM-RULES")), /<system-instructions>/);
});

test("every queued user message is sent, in order", async () => {
	const handle = makeProvider([agent()]);
	const message = await step(handle, [user("ECHO first"), user("second"), user("third")]);
	assert.match(textOf(message), /ECHO first\s*second\s*third>>>/);
});

test("history the session has not seen is sent as a transcript", async () => {
	const handle = makeProvider([agent()]);
	const other: Message = {
		role: "assistant",
		content: [{ type: "text", text: "answer from another model" }],
		api: "x",
		provider: "other",
		model: "m",
		usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
		stopReason: "stop",
		timestamp: ++clock,
	} as Message;
	const text = textOf(await step(handle, [user("earlier question"), other, user("ECHO now")]));
	assert.match(text, /<conversation-history>[\s\S]*\[user\]\nearlier question[\s\S]*\[assistant\]\nanswer from another model[\s\S]*<\/conversation-history>ECHO now/);
});

test("a continued conversation sends only the new message; a rewritten one starts a new session", async () => {
	const handle = makeProvider([agent()]);
	const messages: Message[] = [user("ECHO one")];
	const first = await step(handle, messages);
	const session = /ECHO\[(.+?)\]/.exec(textOf(first))?.[1];
	messages.push(first, user("ECHO two"));
	const second = textOf(await step(handle, messages));
	assert.equal(/ECHO\[(.+?)\]/.exec(second)?.[1], session);
	assert.doesNotMatch(second, /conversation-history/);
	// A tree jump back to the start: the session's reply is gone from Pi's history.
	const branched = textOf(await step(handle, [messages[0]!, { ...(first as AssistantMessage), timestamp: ++clock }, user("ECHO other")]));
	assert.notEqual(/ECHO\[(.+?)\]/.exec(branched)?.[1], session);
	assert.match(branched, /<conversation-history>[\s\S]*ECHO one/);
});

test("steering reaches the running turn with _session/steering before the tool result", async () => {
	const handle = makeProvider([agent({ env: { FAKE_STEERING: "1" } })]);
	const messages: Message[] = [user('MCPCALL:echo:{"text":"hi"}')];
	const first = await step(handle, messages);
	messages.push(first, ...(await answerTools(handle, first)), user("STEER-NOW"));
	const last = textOf(await step(handle, messages));
	assert.match(last, /RESULT<<<echoed hi>>> STEERED<<<STEER-NOW>>>/);
});

test("without steering support, a steer rides on the Pi tool result", async () => {
	const handle = makeProvider([agent()]);
	const messages: Message[] = [user('MCPCALL:echo:{"text":"hi"}')];
	const first = await step(handle, messages);
	messages.push(first, ...(await answerTools(handle, first)), user("STEER-NOW"));
	const last = textOf(await step(handle, messages));
	assert.match(last, /RESULT<<<echoed hi\|<user-message>[\s\S]*STEER-NOW[\s\S]*<\/user-message>>>>/);
	assert.doesNotMatch(last, /STEERED/);
});

test("a steer after the agent's own tool call is sent as the next prompt once the turn ends", async () => {
	const handle = makeProvider([agent({ piTools: "off" })], { nativeTools: () => true });
	const messages: Message[] = [user("hello")];
	const first = await step(handle, messages);
	assert.equal(first.stopReason, "toolUse");
	messages.push(first, ...(await answerTools(handle, first)), user("ECHO follow-up"));
	const last = await step(handle, messages);
	assert.equal(last.stopReason, "stop");
	assert.match(textOf(last), /Hello world #0[\s\S]*ECHO\[sess-1\]<<<ECHO follow-up>>>/);
});

test("isolated requests run in a separate agent process", async () => {
	const handle = makeProvider([agent({ piTools: "off" })]);
	const messages: Message[] = [user("ECHO main")];
	const first = await step(handle, messages);
	const summary = textOf(await step(handle, [user("ECHO summarize")], "SUMMARIZER", { cacheRetention: "none" }));
	assert.match(summary, /SUMMARIZER[\s\S]*ECHO summarize/);
	messages.push(first, user("ECHO again"));
	const again = textOf(await step(handle, messages));
	assert.match(again, /ECHO\[sess-1\]<<<ECHO again>>>/, "the main session is untouched");
});

test("cache-warming requests do not reach the agent", async () => {
	const handle = makeProvider([agent({ command: "/nonexistent/agent" })]);
	const message = await step(handle, [user("hi")], undefined, { maxTokens: 1, disableContinuation: true });
	assert.equal(message.stopReason, "stop");
	assert.deepEqual(message.content, []);
});

test("top-level combinators are flattened for MCP, with the constraint in the description", () => {
	const { schema, note } = flattenRootCombinators({
		type: "object",
		properties: { mode: { type: "string" } },
		oneOf: [
			{ properties: { path: { type: "string" } }, required: ["path"] },
			{ properties: { url: { type: "string" } }, required: ["url"] },
		],
	});
	assert.deepEqual(schema, { type: "object", properties: { mode: { type: "string" }, path: { type: "string" }, url: { type: "string" } } });
	assert.equal(note, "Input constraint: provide parameters for exactly one of: (path) or (url).");
	assert.equal(toMcpTool({ name: "bad", description: "", parameters: { type: "string" } }), null);
	assert.deepEqual(toMcpTool(echoTool)?.inputSchema, { type: "object", properties: { text: { type: "string" } }, required: ["text"] });
});

test("Pi tool calls are recognized in each agent's naming", () => {
	assert.equal(piToolRef("mcp__pi__echo_secret", undefined), "echo_secret");
	assert.equal(piToolRef(undefined, "mcp.pi.echo_secret"), "echo_secret");
	assert.equal(piToolRef(undefined, "mcp_pi.echo_secret"), "echo_secret");
	assert.equal(piToolRef("mcp_pi_echo_secret", "Run mcp_pi_echo_secret"), "echo_secret");
	assert.equal(piToolRef(undefined, "pi/read"), "read");
	for (const title of ["Read pipeline.ts", "mcp__pipeline__run", "api/read", "Run pip install", "Searching for relevant tools"]) {
		assert.equal(piToolRef(undefined, title), undefined, title);
	}
});
