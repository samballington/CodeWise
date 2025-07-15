import React, { useState } from 'react';
import { FileText, ChevronRight, Loader2 } from 'lucide-react';

interface SummaryPanelProps {
  projectName: string;
  filePath?: string;
  directoryPath?: string;
  onClose: () => void;
}

export const SummaryPanel: React.FC<SummaryPanelProps> = ({
  projectName,
  filePath,
  directoryPath,
  onClose
}) => {
  const [summary, setSummary] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [query, setQuery] = useState<string>('');

  const fetchSummary = async (customQuery?: string) => {
    setIsLoading(true);
    setError('');
    
    try {
      const queryParam = customQuery || query || '';
      let url: string;
      
      if (directoryPath) {
        url = `/api/projects/${projectName}/dir/${directoryPath}?query=${encodeURIComponent(queryParam)}`;
      } else {
        url = `/api/projects/${projectName}/summary?query=${encodeURIComponent(queryParam)}`;
      }
      
      const response = await fetch(`http://localhost:8000${url}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch summary: ${response.statusText}`);
      }
      
      const data = await response.json();
      setSummary(data.summary || 'No summary available');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch summary');
    } finally {
      setIsLoading(false);
    }
  };

  const handleExplainMore = () => {
    const explainQuery = `explain in detail ${directoryPath || projectName}`;
    fetchSummary(explainQuery);
  };

  React.useEffect(() => {
    fetchSummary();
  }, [projectName, directoryPath]);

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[80vh] overflow-hidden">
        <div className="flex items-center justify-between p-4 border-b">
          <div className="flex items-center space-x-2">
            <FileText className="w-5 h-5 text-blue-600" />
            <h2 className="text-lg font-semibold">
              {directoryPath ? `Directory: ${directoryPath}` : `Project: ${projectName}`}
            </h2>
          </div>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 text-xl font-bold"
          >
            ×
          </button>
        </div>
        
        <div className="p-4 space-y-4">
          {/* Query Input */}
          <div className="flex space-x-2">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a specific question about this code..."
              className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              onKeyPress={(e) => e.key === 'Enter' && fetchSummary()}
            />
            <button
              onClick={() => fetchSummary()}
              disabled={isLoading}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 flex items-center space-x-2"
            >
              {isLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <ChevronRight className="w-4 h-4" />
              )}
              <span>Ask</span>
            </button>
          </div>

          {/* Summary Content */}
          <div className="bg-gray-50 rounded-lg p-4 min-h-[200px] max-h-[400px] overflow-y-auto">
            {isLoading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="w-6 h-6 animate-spin text-blue-600" />
                <span className="ml-2 text-gray-600">Generating summary...</span>
              </div>
            ) : error ? (
              <div className="text-red-600 text-center py-8">
                <p>Error: {error}</p>
                <button
                  onClick={() => fetchSummary()}
                  className="mt-2 text-blue-600 hover:underline"
                >
                  Try again
                </button>
              </div>
            ) : summary ? (
              <div className="prose prose-sm max-w-none">
                <pre className="whitespace-pre-wrap text-sm text-gray-800 font-sans">
                  {summary}
                </pre>
              </div>
            ) : (
              <div className="text-gray-500 text-center py-8">
                No summary available
              </div>
            )}
          </div>

          {/* Action Buttons */}
          <div className="flex justify-between items-center pt-4 border-t">
            <button
              onClick={handleExplainMore}
              disabled={isLoading}
              className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 flex items-center space-x-2"
            >
              <FileText className="w-4 h-4" />
              <span>Explain More</span>
            </button>
            
            <div className="flex space-x-2">
              <button
                onClick={() => fetchSummary('')}
                disabled={isLoading}
                className="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 disabled:opacity-50"
              >
                Refresh
              </button>
              <button
                onClick={onClose}
                className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}; 