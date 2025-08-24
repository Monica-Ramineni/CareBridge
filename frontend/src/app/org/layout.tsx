"use client";
// Auth temporarily disabled in production; remove next-auth buttons
import { usePathname } from "next/navigation";
import Link from "next/link";

export default function OrgLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const isPatient = pathname?.startsWith("/org/patient");
  const isProvider = pathname?.startsWith("/org/provider");
  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-blue-50 to-indigo-100">
      <header className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center text-xl font-semibold text-gray-900">
          <img src="/carebridge-logo-white.png?v=1" alt="CareBridge" width={80} height={80} className="mr-3 h-20 w-20 object-contain select-none" draggable={false} />
          <div className="leading-tight">
            <div className="text-blue-700 tracking-tight">CareBridge</div>
            <div className="text-[11px] text-gray-500">Enterprise Edition</div>
          </div>
        </div>
        <nav className="space-x-2 text-sm">
          <Link href="/" className="px-3 py-2 rounded hover:bg-gray-100 text-gray-700">Guest Chat</Link>
          <Link href="/org/patient" className={`px-3 py-2 rounded ${isPatient ? 'bg-blue-600 text-white hover:bg-blue-700' : 'hover:bg-gray-100 text-gray-700'}`}>Patient</Link>
          <Link href="/org/provider" className={`px-3 py-2 rounded ${isProvider ? 'bg-blue-600 text-white hover:bg-blue-700' : 'hover:bg-gray-100 text-gray-700'}`}>Provider</Link>
        </nav>
        <div className="ml-auto pl-4 flex items-center gap-2">
          <span className="hidden sm:inline px-2 py-1 text-xs rounded-full bg-gray-100 text-gray-700 border border-gray-200">
            No PHI stored
          </span>
          {/* Profile menu (SSO-gated) */}
          {process.env.NEXT_PUBLIC_ENABLE_SSO === 'true' && (
            <details className="relative">
              <summary className="list-none cursor-pointer rounded-full w-8 h-8 bg-gray-100 border border-gray-200 flex items-center justify-center text-gray-700 hover:bg-gray-200" aria-label="Profile">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5"><path fillRule="evenodd" d="M12 2.25a5.25 5.25 0 100 10.5 5.25 5.25 0 000-10.5zM4.5 20.25a7.5 7.5 0 0115 0v.75a.75.75 0 01-.75.75h-13.5a.75.75 0 01-.75-.75v-.75z" clipRule="evenodd" /></svg>
              </summary>
              <div className="absolute right-0 mt-2 w-44 bg-white border border-gray-200 rounded shadow-md z-10 text-sm">
                <a href="/api/auth/signin/github?callbackUrl=/org/patient" className="block px-3 py-2 hover:bg-gray-50">Sign in with GitHub</a>
                <div className="border-t border-gray-200" />
                <a href="/org/settings" className="block px-3 py-2 hover:bg-gray-50">Settings</a>
                <a href="/api/auth/signout?callbackUrl=/" className="block px-3 py-2 hover:bg-gray-50">Sign out</a>
              </div>
            </details>
          )}
        </div>
      </header>
      <main className="flex-1">{children}</main>
    </div>
  );
}


