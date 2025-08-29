import React from 'react';

interface FallbackDisplayProps {
  message: string;
  error?: Error;
}

export default function FallbackDisplay({ message, error }: FallbackDisplayProps) {
  // In a real application, you would log the error to a monitoring service.
  if (error) {
    console.error("Caught by FallbackDisplay:", error);
  }

  return (
    <div className="my-2 border-l-4 border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20 dark:border-yellow-400 p-4 text-yellow-800 dark:text-yellow-200">
      <p className="font-semibold">System Message</p>
      <p>{message}</p>
      {error && (
        <details className="mt-2">
          <summary className="cursor-pointer text-sm opacity-70">Error Details</summary>
          <pre className="mt-1 text-xs opacity-60 whitespace-pre-wrap">{error.message}</pre>
        </details>
      )}
    </div>
  );
}