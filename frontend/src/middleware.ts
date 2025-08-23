// Guarded middleware: when SSO flag is ON, delegate to next-auth middleware; otherwise allow
import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";
import nextAuth from "next-auth/middleware";

export const config = {
  matcher: ["/org/:path*"],
};

export default function middleware(req: NextRequest) {
  const enabled = process.env.NEXT_PUBLIC_ENABLE_SSO === "true";
  if (!enabled) return NextResponse.next();
  // Enforce auth via next-auth's middleware
  // @ts-expect-error - next-auth middleware has default export signature (req) => NextResponse
  return nextAuth(req);
}


