// turndown ships no types and there is no @types/turndown package.
// Minimal shim covering the surface used by extractText().
declare module "turndown" {
  interface TurndownNode {
    tagName?: string;
  }
  interface TurndownOptions {
    headingStyle?: "setext" | "atx";
    codeBlockStyle?: "indented" | "fenced";
    blankReplacement?: (content: string, node: TurndownNode) => string;
  }
  class TurndownService {
    constructor(options?: TurndownOptions);
    remove(filters: string | string[]): this;
    turndown(html: string): string;
  }
  export default TurndownService;
}
