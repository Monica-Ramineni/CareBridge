"use client";

// Auth disabled: render children directly without SessionProvider
export default function Providers({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}


