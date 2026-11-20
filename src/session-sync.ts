/**
 * What an ACP session has seen of the Pi conversation, and the prompt that sends it the rest.
 *
 * Nothing here talks to an agent: `buildPromptBlocks` assembles a prompt from a snapshot, and its
 * `commit` records what the prompt delivered once the agent accepts it.
 */

import type { AssistantMessage, Message, UserMessage } from "@earendil-works/pi-ai";
import { isRecord } from "./acp.ts";
import type { SystemPromptMode } from "./config.ts";
import { ACP_PROVIDER_ID } from "./models.ts";
import { messageKey, tagged, transcriptOf, userBlocks, type AcpContentBlock } from "./transcript.ts";

/** Rejections after which a steer is dropped rather than sent again. */
export const MAX_STEER_REJECTIONS = 2;

/** Put in front of user messages that reach the agent with a later prompt than the one meant to carry them. */
export const LATE_STEER_NOTE = "Delivered late: the user sent the following earlier, and it did not reach you then.";

/** What the provider knows one ACP session has seen of the Pi conversation. */
export interface SyncState {
	/** Timestamps of the assistant messages this session streamed. */
	ownStamps: Set<number>;
	/** The newest of those seen in Pi's history: what the session has seen goes up to here. */
	anchor: number | undefined;
	/** Key of the first history message; compaction replaces it. */
	first: string | undefined;
	/** Tool call ids this session produced (mirrored and Pi calls): their results are not news to it. */
	ours: Set<string>;
	/** Keys of user messages already delivered (prompted, steered, or attached to a tool result). */
	delivered: Set<string>;
	/**
	 * Keys of history messages the session accepted in a `<conversation-history>` seed. Kept apart from
	 * `anchor`, which moves only past the session's own replies, so a seed of foreign history is not
	 * sent again while no own reply follows it.
	 */
	seeded: Set<string>;
	/**
	 * Timestamp of the newest reply the session streamed to an end Pi keeps in context. Unlike `anchor`,
	 * it covers the latest reply, so a fork that keeps an earlier one is still seen.
	 * WHY: `error` replies are left out because Pi omits a retried one from context; `aborted` ones it keeps.
	 */
	lastStreamed: number | undefined;
	/** Keys of the `seeded` messages the accepted seeds rendered: the ones the session actually read. */
	shown: Set<string>;
	/**
	 * Keys of delivered users and `shown` messages that a later request still held, so Pi's history keeps
	 * them; one missing means the history was edited.
	 * WHY: a Pi `context` handler may add a message of any role for one request only, and nothing tells it
	 * from persisted history, so a message counts only once it is seen again. An edit that drops one before
	 * that goes unnoticed.
	 */
	kept: Set<string>;
	/** The Pi system prompt the session last received. */
	systemPrompt: string;
}

export const newSyncState = (systemPrompt: string): SyncState => ({
	ownStamps: new Set(),
	anchor: undefined,
	first: undefined,
	ours: new Set(),
	delivered: new Set(),
	seeded: new Set(),
	lastStreamed: undefined,
	shown: new Set(),
	kept: new Set(),
	systemPrompt,
});

/** Index after the last assistant message: where the messages Pi added since start. */
export function tailStart(messages: readonly Message[]): number {
	for (let index = messages.length - 1; index >= 0; index--) if (messages[index]?.role === "assistant") return index + 1;
	return 0;
}

/**
 * Index after the last assistant message that answered: where the users of the request Pi makes start.
 * An assistant message that ended in `error` answered nothing, so the users before it still wait for a
 * reply; the ones the session already got are filtered out as delivered. An `aborted` one counts as an
 * answer: the user stopped that request, so it must not run again.
 */
function requestStart(messages: readonly Message[]): number {
	for (let index = messages.length - 1; index >= 0; index--) {
		const candidate = messages[index];
		if (candidate?.role === "assistant" && candidate.stopReason !== "error") return index + 1;
	}
	return 0;
}

/** The user messages of the request Pi makes. */
export function requestUsers(messages: readonly Message[]): UserMessage[] {
	return messages.slice(requestStart(messages)).filter((candidate): candidate is UserMessage => candidate.role === "user");
}

/** Whether the session streamed `message` itself. */
export function isOwnReply(state: SyncState, message: Message): boolean {
	return message.role === "assistant" && message.provider === ACP_PROVIDER_ID && state.ownStamps.has(message.timestamp);
}

/** Whether the session produced `message`, or already received it. */
function isOwn(state: SyncState, message: Message): boolean {
	if (isOwnReply(state, message) || state.seeded.has(messageKey(message))) return true;
	if (message.role === "toolResult") return state.ours.has(message.toolCallId);
	return message.role === "user" && state.delivered.has(messageKey(message));
}

/**
 * Whether Pi's history no longer holds what the session saw: the first message
 * changed (compaction), or a tree jump, fork or context edit dropped the
 * session's latest reply, the anchor, or accepted history in `kept`. Messages an
 * extension adds for one request only do not count.
 */
export function diverged(state: SyncState, messages: readonly Message[]): boolean {
	if (state.first !== undefined && messages[0] && messageKey(messages[0]) !== state.first) return true;
	const hasReply = (timestamp: number): boolean =>
		messages.some((candidate) => candidate.role === "assistant" && candidate.timestamp === timestamp);
	if (state.anchor !== undefined && !hasReply(state.anchor)) return true;
	if (state.lastStreamed !== undefined && !hasReply(state.lastStreamed)) return true;
	const keys = new Set(messages.map(messageKey));
	return [...state.kept].some((key) => !keys.has(key));
}

/** Adds to `kept` what earlier requests delivered or showed the session that Pi's history still holds. */
export function keepConfirmed(state: SyncState, messages: readonly Message[]): void {
	for (const message of messages) {
		const key = messageKey(message);
		if (state.delivered.has(key) || state.shown.has(key)) state.kept.add(key);
	}
}

/** Records a settled Pi message as the session's latest reply, if the session streamed it and Pi keeps it in context. */
export function noteStreamed(state: SyncState | undefined, message: AssistantMessage): void {
	if (!state?.ownStamps.has(message.timestamp) || message.stopReason === "error") return;
	state.lastStreamed = message.timestamp;
}

/** A steering message no prompt the agent accepted has carried yet. */
export interface PendingSteer {
	message: UserMessage;
	key: string;
	/** How many prompts carrying it the agent rejected. */
	rejections: number;
}

export const isDropped = (steer: PendingSteer): boolean => steer.rejections >= MAX_STEER_REJECTIONS;

/**
 * Steers the agent has not accepted yet, and users an `onPayload` hook left out of an accepted prompt,
 * oldest first, by owner (`turnKey` of agent and conversation).
 * WHY: kept apart from any `SyncState`, so they outlive a replaced or re-seeded session and reach its next prompt.
 */
export class PendingSteers {
	readonly #byOwner = new Map<string, PendingSteer[]>();

	has(owner: string, key: string): boolean {
		return this.#byOwner.get(owner)?.some((steer) => steer.key === key) === true;
	}

	/** Holds `steers` until a prompt carrying them is accepted. */
	hold(owner: string, steers: readonly UserMessage[]): void {
		const held = this.#byOwner.get(owner) ?? [];
		for (const message of steers) {
			const key = messageKey(message);
			if (!held.some((steer) => steer.key === key)) held.push({ message, key, rejections: 0 });
		}
		this.#byOwner.set(owner, held);
	}

	/** Records the steers `keys` as delivered to the session. */
	deliver(owner: string, state: SyncState | undefined, keys: readonly string[]): void {
		for (const key of keys) state?.delivered.add(key);
		const held = this.#byOwner.get(owner);
		if (held) this.#byOwner.set(owner, held.filter((steer) => !keys.includes(steer.key)));
	}

	/** Counts a rejection of the steers `keys`; returns the ones this rejection dropped. */
	reject(owner: string, keys: readonly string[]): PendingSteer[] {
		const dropped: PendingSteer[] = [];
		for (const steer of this.#byOwner.get(owner) ?? []) {
			if (!keys.includes(steer.key)) continue;
			steer.rejections++;
			if (steer.rejections === MAX_STEER_REJECTIONS) dropped.push(steer);
		}
		return dropped;
	}

	/**
	 * The pending steers of `owner` still on Pi's active branch and undelivered to the session; forgets the rest.
	 * WARN: dropped steers stay listed, unsent, so that they stay out of history seeds as well.
	 */
	recover(owner: string, messages: readonly Message[], state: SyncState): PendingSteer[] {
		const branch = new Set(messages.map(messageKey));
		const kept = (this.#byOwner.get(owner) ?? []).filter((steer) => branch.has(steer.key) && !state.delivered.has(steer.key));
		this.#byOwner.set(owner, kept);
		return kept;
	}

	clear(): void {
		this.#byOwner.clear();
	}
}

/** Blocks of a prompt and what sending them tells about the session, once the agent accepts it. */
export interface PromptPart {
	blocks: AcpContentBlock[];
	/** The blocks as sent, fixed before an `onPayload` hook can change them. */
	sent: string[];
	record: () => void;
	/** What the prompt leaving the part out means, once the agent accepts it. */
	omit: () => void;
	/** Key of the pending steer the part carries. */
	steer: string | undefined;
}

function promptPart(blocks: AcpContentBlock[], record: () => void, options: { omit?: () => void; steer?: string } = {}): PromptPart {
	return { blocks, sent: blocks.map((block) => JSON.stringify(block)), record, omit: options.omit ?? (() => {}), steer: options.steer };
}

/**
 * The parts whose blocks all reach the agent in `payload`, which an `onPayload` hook may have replaced
 * or changed. A block the hook kept is recognized as the same object, unchanged. A block it rebuilt is
 * matched by value, but only when no other block of the prompt has that value, since it cannot tell
 * which of two identical blocks it stands for.
 * WARN: a part a hook rewrote, or an ambiguous one, counts as not sent: wrongly sending again is
 * recoverable, wrongly recording delivery loses the content.
 */
export function partsIn(parts: readonly PromptPart[], payload: unknown): PromptPart[] {
	const prompt: unknown[] = isRecord(payload) && Array.isArray(payload.prompt) ? payload.prompt : [];
	const ours = new Set<unknown>(parts.flatMap((part) => part.blocks));
	const kept = new Set<unknown>();
	const rebuilt = new Map<string, number>();
	for (const block of prompt) {
		if (ours.has(block)) kept.add(block);
		else {
			const text = JSON.stringify(block);
			rebuilt.set(text, (rebuilt.get(text) ?? 0) + 1);
		}
	}
	const owners = new Map<string, number>();
	for (const text of parts.flatMap((part) => part.sent)) owners.set(text, (owners.get(text) ?? 0) + 1);
	return parts.filter((part) =>
		part.blocks.every((block, index) => {
			const text = part.sent[index]!;
			if (kept.has(block)) return JSON.stringify(block) === text;
			return owners.get(text) === 1 && rebuilt.get(text) === 1;
		}),
	);
}

export interface PromptInput {
	/** The agent's `systemPrompt` setting. */
	systemPromptMode: SystemPromptMode;
	/** Pi's current system prompt. */
	systemPrompt: string;
	state: SyncState;
	/** Pi's conversation (`conversation(context.messages)`). */
	messages: readonly Message[];
	/** The pending steers `PendingSteers.recover` returned for this request. */
	held: readonly PendingSteer[];
	/** Whether the agent takes image blocks. */
	images: boolean;
	/** Timestamp of the Pi message the prompt streams into. */
	timestamp: number;
	steers: PendingSteers;
	/** The `PendingSteers` owner of the request. */
	owner: string;
}

export interface PromptBuild {
	parts: PromptPart[];
	blocks: AcpContentBlock[];
	/**
	 * Records what the prompt delivered, given the parts the sent payload carried (`partsIn`).
	 * WARN: call it only once the agent accepts the prompt; until then nothing here has changed any state.
	 */
	commit: (sent: readonly PromptPart[]) => void;
}

/**
 * The prompt that brings the session up to date with Pi's conversation: changed system instructions, a
 * `<conversation-history>` seed of what it has not seen, late steers, then the request's new users.
 * Null when there is no user message left to send.
 */
export function buildPromptBlocks(input: PromptInput): PromptBuild | null {
	const { systemPromptMode, systemPrompt, state, messages, held, images, timestamp, steers, owner } = input;
	const tail = messages.slice(tailStart(messages));
	const users = requestUsers(messages);
	const parts: PromptPart[] = [];
	if (systemPromptMode !== "off" && systemPrompt !== "" && state.systemPrompt !== systemPrompt) {
		const intro = state.systemPrompt
			? "Pi, the host application you run in, updated its instructions. They replace the earlier ones:"
			: "Instructions from Pi, the host application you run in. Follow them together with your own:";
		parts.push(promptPart([tagged("system-instructions", systemPrompt, intro)], () => (state.systemPrompt = systemPrompt)));
	}
	const isHeld = (candidate: Message): boolean => held.some((steer) => steer.key === messageKey(candidate));
	const late = held.filter((steer) => !isDropped(steer));
	const requestUserSet = new Set<Message>(users);
	const history = messages.slice(0, messages.length - tail.length);
	const anchorIndex = state.anchor === undefined ? -1 : history.findLastIndex((candidate) => candidate.role === "assistant" && candidate.timestamp === state.anchor);
	// WHY: pending steers and the users of an unanswered request are requests the agent never got, so
	// they go as user blocks, never as history.
	const unseen = history
		.slice(anchorIndex + 1)
		.filter((candidate) => !isOwn(state, candidate) && !isHeld(candidate) && !requestUserSet.has(candidate));
	/** How much of `history` the session holds once the prompt is accepted. */
	let heldHistory = history.length;
	const { text: transcript, shown } = transcriptOf(unseen);
	if (transcript) {
		const intro = "Earlier messages of this Pi conversation that you have not seen, oldest first. Treat them as history, not as new requests:";
		parts.push(
			promptPart(
				[tagged("conversation-history", transcript, intro)],
				() => {
					for (const seen of unseen) state.seeded.add(messageKey(seen));
					for (const seen of shown) state.shown.add(messageKey(seen));
				},
				// WHY: the anchor must stay before history the session never got, or later prompts skip it.
				{ omit: () => (heldHistory = history.indexOf(unseen[0]!)) },
			),
		);
	}
	if (late.length > 0) parts.push(promptPart([{ type: "text", text: LATE_STEER_NOTE }], () => {}));
	for (const steer of late) {
		const record = (): void => steers.deliver(owner, state, [steer.key]);
		parts.push(promptPart(userBlocks(steer.message, images), record, { steer: steer.key }));
	}
	const sendable = users.filter((user) => !isHeld(user));
	const fresh = sendable.filter((user) => !state.delivered.has(messageKey(user)));
	// WHY: with nothing new, the request is Pi retrying the last one; a recovered steer is new enough.
	const sentUsers = fresh.length > 0 || late.length > 0 ? fresh : sendable.slice(-1);
	if (sentUsers.length === 0 && late.length === 0) return null;
	for (const user of sentUsers) {
		// WHY: once the agent answers, a user the hook left out falls behind the reply and would reach the
		// next prompt only as history; held, it goes as a late user block instead.
		const omit = (): void => steers.hold(owner, [user]);
		parts.push(promptPart(userBlocks(user, images), () => state.delivered.add(messageKey(user)), { omit }));
	}
	const commit = (sent: readonly PromptPart[]): void => {
		for (const part of parts) (sent.includes(part) ? part.record : part.omit)();
		const newestOwn = history.slice(0, heldHistory).findLast((candidate) => isOwnReply(state, candidate));
		if (newestOwn) state.anchor = newestOwn.timestamp;
		state.first = messages[0] ? messageKey(messages[0]) : undefined;
		state.ownStamps.add(timestamp);
	};
	return { parts, blocks: parts.flatMap((part) => part.blocks), commit };
}
