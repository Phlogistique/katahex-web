// What the page and the node driver that runs it agree on: the page pulls its
// work one item at a time and hands each result back, so a run that is stopped
// picks up where it left off. Both functions are installed by playwright
// (page.exposeFunction), and the page waits for `done` on the log.

export type Bridge<Job, Result> = {
  nextJob(): Promise<Job | null>;
  report(result: Result): Promise<void>;
  note(line: string): Promise<void>;
};

export function bridge<Job, Result>(): { driver: Bridge<Job, Result>; log: (line: string) => void } {
  const driver = globalThis as unknown as Bridge<Job, Result>;
  const element = document.getElementById('log')!;
  return {
    driver,
    log: (line: string) => {
      element.textContent += line + '\n';
      element.scrollTop = element.scrollHeight;
      void driver.note?.(line);
    },
  };
}
