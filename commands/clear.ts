import { getDbConn } from "../db.ts";
import * as repo from "../repository.ts";
import type { RagCommandHandler } from "./types.ts";

export const clearCommand: RagCommandHandler = ({ ctx }) => {
  repo.wipeIndex(getDbConn());
  ctx.ui.notify("Index cleared.", "info");
};
