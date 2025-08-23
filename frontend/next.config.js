/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    const isDev = process.env.NODE_ENV !== 'production';
    const hasExplicitApiBase = Boolean(process.env.NEXT_PUBLIC_API_BASE);
    if (isDev && !hasExplicitApiBase) {
      // In local dev, proxy only backend endpoints to FastAPI on 8000.
      // Keep NextAuth (/api/auth) and any Next API routes untouched.
      return [
        { source: '/api/chat', destination: 'http://localhost:8000/chat' },
        { source: '/api/health', destination: 'http://localhost:8000/health' },
        { source: '/api/generate-title', destination: 'http://localhost:8000/generate-title' },
        { source: '/api/upload-lab-results', destination: 'http://localhost:8000/upload-lab-results' },
        { source: '/api/lab-results/:path*', destination: 'http://localhost:8000/lab-results/:path*' },
        { source: '/api/v1/:path*', destination: 'http://localhost:8000/v1/:path*' },
        { source: '/api/metrics', destination: 'http://localhost:8000/metrics' },
      ];
    }
    return [];
  },
};

module.exports = nextConfig;


