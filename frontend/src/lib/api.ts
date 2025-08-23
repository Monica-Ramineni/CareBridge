// Central API base for client-side calls. Uses env on Vercel; falls back locally.
// Default to same-origin serverless function namespace on Vercel
export const API_BASE: string =
  (process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "/api");

export const apiUrl = (path: string): string => {
  if (!path) return API_BASE;
  return `${API_BASE}${path.startsWith('/') ? path : `/${path}`}`;
};


