import { create } from 'zustand';
import { Project, FileNode, FileContent, ProjectState } from '../types/project';

interface ProjectStore extends ProjectState {
  // Actions
  setProjects: (projects: Project[]) => void;
  setCurrentProject: (projectName: string | null) => void;
  setFileTree: (tree: FileNode | null) => void;
  addOpenFile: (file: FileContent) => void;
  removeOpenFile: (path: string) => void;
  setActiveFile: (path: string | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  clearAll: () => void;
  
  // API calls
  fetchProjects: () => Promise<void>;
  fetchFileTree: (projectName: string) => Promise<void>;
  fetchFileContent: (projectName: string, filePath: string) => Promise<void>;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const useProjectStore = create<ProjectStore>((set, get) => ({
  // Initial state
  projects: [],
  currentProject: null,
  fileTree: null,
  openFiles: [],
  activeFile: null,
  loading: false,
  error: null,

  // Actions
  setProjects: (projects) => set({ projects }),
  setCurrentProject: (projectName) => set({ currentProject: projectName }),
  setFileTree: (tree) => set({ fileTree: tree }),
  
  addOpenFile: (file) => {
    const { openFiles } = get();
    const existingIndex = openFiles.findIndex(f => f.path === file.path);
    
    if (existingIndex >= 0) {
      // Update existing file
      const newOpenFiles = [...openFiles];
      newOpenFiles[existingIndex] = file;
      set({ openFiles: newOpenFiles, activeFile: file.path });
    } else {
      // Add new file
      set({ openFiles: [...openFiles, file], activeFile: file.path });
    }
  },
  
  removeOpenFile: (path) => {
    const { openFiles, activeFile } = get();
    const newOpenFiles = openFiles.filter(f => f.path !== path);
    const newActiveFile = activeFile === path 
      ? (newOpenFiles.length > 0 ? newOpenFiles[0].path : null)
      : activeFile;
    
    set({ openFiles: newOpenFiles, activeFile: newActiveFile });
  },
  
  setActiveFile: (path) => set({ activeFile: path }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
  
  clearAll: () => set({
    projects: [],
    currentProject: null,
    fileTree: null,
    openFiles: [],
    activeFile: null,
    loading: false,
    error: null
  }),

  // API calls
  fetchProjects: async () => {
    set({ loading: true, error: null });
    try {
      const response = await fetch(`${API_BASE}/projects`);
      if (!response.ok) {
        throw new Error(`Failed to fetch projects: ${response.statusText}`);
      }
      const data = await response.json();
      set({ projects: data.projects, loading: false });
    } catch (error) {
      set({ error: error instanceof Error ? error.message : 'Failed to fetch projects', loading: false });
    }
  },

  fetchFileTree: async (projectName: string) => {
    set({ loading: true, error: null });
    try {
      const response = await fetch(`${API_BASE}/projects/${encodeURIComponent(projectName)}/tree`);
      if (!response.ok) {
        throw new Error(`Failed to fetch file tree: ${response.statusText}`);
      }
      const tree = await response.json();
      set({ fileTree: tree, loading: false });
    } catch (error) {
      set({ error: error instanceof Error ? error.message : 'Failed to fetch file tree', loading: false });
    }
  },

  fetchFileContent: async (projectName: string, filePath: string) => {
    set({ loading: true, error: null });
    try {
      const response = await fetch(`${API_BASE}/projects/${encodeURIComponent(projectName)}/file?path=${encodeURIComponent(filePath)}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch file content: ${response.statusText}`);
      }
      const fileContent = await response.json();
      get().addOpenFile(fileContent);
      set({ loading: false });
    } catch (error) {
      set({ error: error instanceof Error ? error.message : 'Failed to fetch file content', loading: false });
    }
  }
})); 