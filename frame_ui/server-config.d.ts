/** Node-only configuration used by the authenticated Vite proxy. */
export function backend(): { url: string; configPath: string; found: boolean };
export function token(): string;
/** Hosts this dev server may answer to, from `ui_url`. Empty for a plain checkout. */
export function uiHosts(): string[];
