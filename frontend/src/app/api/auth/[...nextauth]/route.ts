// Guarded NextAuth route: only active when NEXT_PUBLIC_ENABLE_SSO=true and GitHub creds exist
import NextAuth from "next-auth";
import GitHubProvider from "next-auth/providers/github";

const ENABLE_SSO = process.env.NEXT_PUBLIC_ENABLE_SSO === "true";
const GITHUB_ID = process.env.GITHUB_ID || process.env.NEXT_PUBLIC_GITHUB_ID;
const GITHUB_SECRET = process.env.GITHUB_SECRET || process.env.NEXT_PUBLIC_GITHUB_SECRET;
const NEXTAUTH_SECRET = process.env.NEXTAUTH_SECRET;

const handler =
  ENABLE_SSO && GITHUB_ID && GITHUB_SECRET
    ? NextAuth({
        providers: [
          GitHubProvider({
            clientId: GITHUB_ID,
            clientSecret: GITHUB_SECRET,
          }),
        ],
        session: { strategy: "jwt" },
        secret: NEXTAUTH_SECRET,
      })
    : (async () => new Response(null, { status: 204 }));

export { handler as GET, handler as POST };


