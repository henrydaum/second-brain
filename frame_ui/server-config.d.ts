/** Node-only configuration used by the authenticated Vite proxy. */
export function backend(): { url: string; configPath: string; found: boolean };
export function token(): string;
