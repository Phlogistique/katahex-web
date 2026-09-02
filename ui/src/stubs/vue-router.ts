/*
 * Hexplorer is the whole app here, so there is nothing to route and nothing to leave.
 * The analyzer caches it would persist on leaving are persisted when the app goes to the
 * background instead, in main.ts.
 */
export const onBeforeRouteLeave = (_guard: unknown): void => {};
