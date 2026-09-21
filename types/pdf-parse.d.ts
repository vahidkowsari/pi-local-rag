// pdf-parse ships no types. The inner path bypasses the index.js debug-mode
// side effect (which tries to read a missing test PDF when imported as main).
declare module "pdf-parse/lib/pdf-parse.js" {
  const pdf: (
    data: Buffer,
    options?: {
      pagerender?: (pageData: { getTextContent: () => Promise<{ items: Array<{ str?: string }> }> }) => Promise<string>;
    },
  ) => Promise<{ text: string; numpages?: number }>;
  export default pdf;
}
