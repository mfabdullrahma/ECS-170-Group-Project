export default function NotFound() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-white dark:bg-black">
      <div className="text-center">
        <h1 className="text-4xl font-bold mb-4 text-black dark:text-white">404</h1>
        <p className="text-lg text-black/60 dark:text-white/60 mb-8">Page not found</p>
        <a
          href="/"
          className="px-6 py-3 bg-black dark:bg-white text-white dark:text-black rounded-lg hover:opacity-80 transition-opacity"
        >
          Go back home
        </a>
      </div>
    </div>
  );
}

