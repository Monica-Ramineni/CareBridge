import NextAuth from "next-auth";
import GitHub from "next-auth/providers/github";

// Make providers safe in local dev without GitHub envs
const providers = [] as any[];
if (process.env.GITHUB_ID && process.env.GITHUB_SECRET) {
  providers.push(
    GitHub({
      clientId: process.env.GITHUB_ID,
      clientSecret: process.env.GITHUB_SECRET,
    })
  );
}

const handler = NextAuth({
  providers,
  session: { strategy: "jwt" },
});

export { handler as GET, handler as POST };


