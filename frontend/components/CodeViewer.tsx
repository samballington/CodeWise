import React from 'react';
import Editor from '@monaco-editor/react';
import { useProjectStore } from '../lib/projectStore';
import { X, FileText, AlertCircle } from 'lucide-react';
import { FileIcon } from './FileIcon';

export const CodeViewer: React.FC = () => {
  const { 
    openFiles, 
    activeFile, 
    setActiveFile, 
    removeOpenFile,
    loading 
  } = useProjectStore();

  const activeFileData = openFiles.find(file => file.path === activeFile);

  const getLanguage = (fileName: string, mimeType?: string) => {
    const extension = fileName.split('.').pop()?.toLowerCase();
    
    switch (extension) {
      case 'js':
      case 'jsx':
        return 'javascript';
      case 'ts':
      case 'tsx':
        return 'typescript';
      case 'py':
        return 'python';
      case 'html':
        return 'html';
      case 'css':
        return 'css';
      case 'scss':
      case 'sass':
        return 'scss';
      case 'json':
        return 'json';
      case 'md':
      case 'markdown':
        return 'markdown';
      case 'yaml':
      case 'yml':
        return 'yaml';
      case 'xml':
        return 'xml';
      case 'sql':
        return 'sql';
      case 'sh':
      case 'bash':
        return 'shell';
      case 'dockerfile':
        return 'dockerfile';
      case 'go':
        return 'go';
      case 'rs':
        return 'rust';
      case 'php':
        return 'php';
      case 'rb':
        return 'ruby';
      case 'java':
        return 'java';
      case 'c':
        return 'c';
      case 'cpp':
      case 'cc':
      case 'cxx':
        return 'cpp';
      case 'cs':
        return 'csharp';
      default:
        return 'plaintext';
    }
  };

  const handleTabClose = (filePath: string, e: React.MouseEvent) => {
    e.stopPropagation();
    removeOpenFile(filePath);
  };

  if (openFiles.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center bg-gray-50">
        <div className="text-center text-gray-500">
          <FileText className="w-16 h-16 mx-auto mb-4 text-gray-300" />
          <h3 className="text-lg font-medium mb-2">No files open</h3>
          <p className="text-sm">Select a file from the project tree to view its contents</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full flex flex-col bg-white code-viewer-container">
      {/* File Tabs */}
      <div className="w-full flex border-b border-gray-200 bg-gray-50 overflow-x-auto flex-shrink-0">
        {openFiles.map((file) => (
          <div
            key={file.path}
            className={`
              flex items-center px-3 py-2 border-r border-gray-200 cursor-pointer
              min-w-0 max-w-xs group hover:bg-gray-100 flex-shrink-0
              ${activeFile === file.path 
                ? 'bg-white border-b-2 border-blue-500' 
                : 'bg-gray-50'
              }
            `}
            onClick={() => setActiveFile(file.path)}
          >
            <FileIcon 
              fileName={file.name} 
              fileType="file" 
              className="mr-2 flex-shrink-0"
            />
            <span className="text-sm truncate mr-2">
              {file.name}
            </span>
            <button
              onClick={(e) => handleTabClose(file.path, e)}
              className="opacity-0 group-hover:opacity-100 hover:bg-gray-200 rounded p-1 transition-opacity"
            >
              <X className="w-3 h-3 text-gray-500" />
            </button>
          </div>
        ))}
      </div>

      {/* File Content */}
      <div className="flex-1 w-full relative">
        {loading && (
          <div className="absolute inset-0 bg-white bg-opacity-75 flex items-center justify-center z-10">
            <div className="text-sm text-gray-500">Loading...</div>
          </div>
        )}
        
        {activeFileData ? (
          activeFileData.is_binary ? (
            <div className="flex items-center justify-center h-full w-full">
              <div className="text-center text-gray-500">
                <AlertCircle className="w-16 h-16 mx-auto mb-4 text-gray-300" />
                <h3 className="text-lg font-medium mb-2">Binary File</h3>
                <p className="text-sm">
                  Cannot display binary file: {activeFileData.name}
                </p>
                <p className="text-xs text-gray-400 mt-2">
                  Size: {formatFileSize(activeFileData.size)} • 
                  Type: {activeFileData.mime_type}
                </p>
              </div>
            </div>
          ) : (
            <div className="h-full w-full" style={{width: '100%', minWidth: '100%'}}>
              <Editor
                width="100%"
                height="100%"
                language={getLanguage(activeFileData.name, activeFileData.mime_type)}
                value={activeFileData.content || ''}
                theme="vs-light"
                options={{
                  readOnly: true,
                  minimap: { enabled: false },
                  fontSize: 14,
                  lineNumbers: 'on',
                  scrollBeyondLastLine: false,
                  automaticLayout: true,
                  wordWrap: 'on',
                  folding: true,
                  lineDecorationsWidth: 10,
                  lineNumbersMinChars: 3,
                }}
              />
            </div>
          )
        ) : (
          <div className="flex items-center justify-center h-full w-full">
            <div className="text-center text-gray-500">
              <FileText className="w-16 h-16 mx-auto mb-4 text-gray-300" />
              <h3 className="text-lg font-medium mb-2">File not found</h3>
              <p className="text-sm">The selected file could not be loaded</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const formatFileSize = (bytes: number) => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}; 