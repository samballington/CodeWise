'use client'

import React, { useState } from 'react'
import { ChevronRight, ChevronDown, File, Folder, Home, Info, Trash2, AlertTriangle } from 'lucide-react'
import { FileIcon } from './FileIcon'
import { useThemeStore } from '../lib/store'
import { useProjectStore } from '../lib/projectStore';
import { FileNode } from '../types/project';
import { SummaryPanel } from './SummaryPanel';

interface FileTreeNodeProps {
  node: FileNode;
  level: number;
  onFileClick: (filePath: string) => void;
  onSummaryClick: (path: string, isDirectory: boolean) => void;
}

const FileTreeNode: React.FC<FileTreeNodeProps> = ({ node, level, onFileClick, onSummaryClick }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const hasChildren = node.children && node.children.length > 0;

  const handleClick = () => {
    if (node.type === 'directory') {
      setIsExpanded(!isExpanded);
    } else {
      onFileClick(node.path);
    }
  };

  const handleToggle = (e: React.MouseEvent) => {
    e.stopPropagation();
    setIsExpanded(!isExpanded);
  };

  const handleSummaryClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    onSummaryClick(node.path, node.type === 'directory');
  };

  return (
    <div>
      <div
        className={`
          flex items-center py-1 px-2 hover:bg-gray-100 cursor-pointer
          text-sm select-none group
        `}
        style={{ paddingLeft: `${level * 12 + 8}px` }}
        onClick={handleClick}
      >
        {node.type === 'directory' && (
          <button
            onClick={handleToggle}
            className="flex items-center justify-center w-4 h-4 mr-1 hover:bg-gray-200 rounded"
          >
            {hasChildren && (
              isExpanded ? (
                <ChevronDown className="w-3 h-3 text-gray-500" />
              ) : (
                <ChevronRight className="w-3 h-3 text-gray-500" />
              )
            )}
          </button>
        )}
        
        {node.type === 'file' && (
          <div className="w-4 h-4 mr-1" />
        )}
        
        <FileIcon 
          fileName={node.name} 
          fileType={node.type} 
          isExpanded={isExpanded}
          className="mr-2"
        />
        
        <span className="text-gray-800 truncate">
          {node.name}
        </span>
        
        {/* Summary button */}
        <button
          onClick={handleSummaryClick}
          className="opacity-0 group-hover:opacity-100 p-1 hover:bg-gray-200 rounded transition-opacity ml-auto"
          title="View summary"
        >
          <Info className="w-3 h-3 text-gray-500" />
        </button>
        
        {node.type === 'file' && node.size && (
          <span className="text-xs text-gray-500 ml-2">
            {formatFileSize(node.size)}
          </span>
        )}
      </div>
      
      {node.type === 'directory' && isExpanded && hasChildren && (
        <div>
          {node.children!.map((child, index) => (
            <FileTreeNode
              key={`${child.path}-${index}`}
              node={child}
              level={level + 1}
              onFileClick={onFileClick}
              onSummaryClick={onSummaryClick}
            />
          ))}
        </div>
      )}
    </div>
  );
};

const formatFileSize = (bytes: number) => {
  if (bytes === 0) return '0B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + sizes[i];
};

export const FileTree: React.FC = () => {
  const { 
    currentProject, 
    fileTree, 
    fetchFileContent,
    loading,
    error 
  } = useProjectStore();
  
  const [summaryPanel, setSummaryPanel] = useState<{
    projectName: string;
    directoryPath?: string;
    filePath?: string;
  } | null>(null);

  const [deleteModal, setDeleteModal] = useState<{
    show: boolean;
    projectName: string;
  }>({ show: false, projectName: '' });
  
  const [isDeleting, setIsDeleting] = useState(false);
  const { isDarkMode } = useThemeStore();

  const handleFileClick = async (filePath: string) => {
    if (currentProject) {
      await fetchFileContent(currentProject, filePath);
    }
  };

  const handleSummaryClick = (path: string, isDirectory: boolean) => {
    if (currentProject) {
      setSummaryPanel({
        projectName: currentProject,
        directoryPath: isDirectory ? path : undefined,
        filePath: !isDirectory ? path : undefined,
      });
    }
  };

  const handleDeleteProject = (projectName: string) => {
    setDeleteModal({ show: true, projectName });
  };

  const confirmDeleteProject = async () => {
    if (!deleteModal.projectName) return;
    
    setIsDeleting(true);
    try {
      const response = await fetch(`http://localhost:8000/projects/${deleteModal.projectName}`, {
        method: 'DELETE',
      });
      
      const result = await response.json();
      
      if (response.ok) {
        alert(`Project '${deleteModal.projectName}' deleted successfully!`);
        setDeleteModal({ show: false, projectName: '' });
        // Refresh the page to update project list
        window.location.reload();
      } else {
        alert(`Failed to delete project: ${result.detail}`);
      }
    } catch (error) {
      alert(`Error deleting project: ${error}`);
    } finally {
      setIsDeleting(false);
    }
  };

  const cancelDelete = () => {
    setDeleteModal({ show: false, projectName: '' });
  };

  if (!currentProject) {
    return (
      <div className="p-4 text-center text-gray-500">
        <div className="text-sm">Select a project to view files</div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="p-4">
        <div className="animate-pulse space-y-2">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-gray-200 rounded"></div>
              <div className="h-4 bg-gray-200 rounded flex-1"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 text-center text-red-500">
        <div className="text-sm">Error loading file tree: {error}</div>
      </div>
    );
  }

  if (!fileTree) {
    return (
      <div className="p-4 text-center text-gray-500">
        <div className="text-sm">No file tree available</div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto bg-white">
      <div className="p-2">
        <div className="flex items-center justify-between mb-2 px-2">
          <div className="text-xs font-medium text-gray-500">
            {currentProject}
          </div>
          {currentProject !== 'workspace' && (
            <button
              onClick={() => handleDeleteProject(currentProject)}
              className="opacity-0 hover:opacity-100 p-1 hover:bg-red-100 rounded transition-opacity"
              title="Delete project"
            >
              <Trash2 className="w-3 h-3 text-red-500" />
            </button>
          )}
        </div>
        <FileTreeNode
          node={fileTree}
          level={0}
          onFileClick={handleFileClick}
          onSummaryClick={handleSummaryClick}
        />
      </div>
      
      {/* Delete Confirmation Modal */}
      {deleteModal.show && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className={`bg-white rounded-lg p-6 max-w-md w-full mx-4 ${isDarkMode ? 'bg-background-secondary text-text-primary' : 'bg-white text-gray-900'}`}>
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-red-100 rounded-full">
                <AlertTriangle className="w-6 h-6 text-red-600" />
              </div>
              <div>
                <h3 className="text-lg font-semibold">Delete Project</h3>
                <p className={`text-sm ${isDarkMode ? 'text-text-secondary' : 'text-gray-600'}`}>
                  This action cannot be undone
                </p>
              </div>
            </div>
            
            <p className={`mb-6 ${isDarkMode ? 'text-text-primary' : 'text-gray-700'}`}>
              Are you sure you want to delete the project <strong>"{deleteModal.projectName}"</strong>? 
              This will permanently remove all files and vector embeddings associated with this project.
            </p>
            
            <div className="flex gap-3 justify-end">
              <button
                onClick={cancelDelete}
                disabled={isDeleting}
                className={`px-4 py-2 rounded border ${isDarkMode ? 'border-border text-text-primary hover:bg-background' : 'border-gray-300 text-gray-700 hover:bg-gray-50'}`}
              >
                Cancel
              </button>
              <button
                onClick={confirmDeleteProject}
                disabled={isDeleting}
                className={`px-4 py-2 rounded text-white ${isDeleting ? 'bg-gray-400' : 'bg-red-600 hover:bg-red-700'}`}
              >
                {isDeleting ? 'Deleting...' : 'Delete Project'}
              </button>
            </div>
          </div>
        </div>
      )}
      
      {summaryPanel && (
        <SummaryPanel
          projectName={summaryPanel.projectName}
          directoryPath={summaryPanel.directoryPath}
          filePath={summaryPanel.filePath}
          onClose={() => setSummaryPanel(null)}
        />
      )}
    </div>
  );
}; 