// Auth temporarily disabled in production build to avoid /api/auth errors on Vercel.
// Keeping this file returns 204s so clients don't error if they still call /api/auth/*.
import { NextResponse } from "next/server";

export async function GET() {
  return new NextResponse(null, { status: 204 });
}

export async function POST() {
  return new NextResponse(null, { status: 204 });
}


