import React, { useState } from 'react';
import { ProjectSelector } from './ProjectSelector';
import { FileTree } from './FileTree';
import { CodeViewer } from './CodeViewer';
import { PanelLeft, PanelLeftClose } from 'lucide-react';

export const ProjectLayout: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <div className="flex flex-1 min-h-0 bg-gray-100">
      {/* Sidebar */}
      <div className={`
        ${sidebarOpen ? 'w-64' : 'w-0'} 
        transition-all duration-300 ease-in-out
        bg-white border-r border-gray-200 flex flex-col
        overflow-hidden
      `}>
        <ProjectSelector />
        <FileTree />
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 project-layout-main">
        {/* Header */}
        <div className="h-12 bg-white border-b border-gray-200 flex items-center px-4 flex-shrink-0">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 hover:bg-gray-100 rounded-md transition-colors"
            title={sidebarOpen ? "Hide sidebar" : "Show sidebar"}
          >
            {sidebarOpen ? (
              <PanelLeftClose className="w-5 h-5 text-gray-600" />
            ) : (
              <PanelLeft className="w-5 h-5 text-gray-600" />
            )}
          </button>
          
          <div className="ml-4">
            <h1 className="text-lg font-semibold text-gray-800">
              CodeWise Project Explorer
            </h1>
          </div>
        </div>

        {/* Code Viewer */}
        <div className="flex-1 min-w-0 w-full">
          <CodeViewer />
        </div>
      </div>
    </div>
  );
}; 