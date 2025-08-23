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
          <span className="text-blue-700 tracking-tight">CareBridge</span>
          <span className="ml-1">— Enterprise</span>
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
          {/* Auth controls disabled for now */}
        </div>
      </header>
      <main className="flex-1">{children}</main>
    </div>
  );
}


