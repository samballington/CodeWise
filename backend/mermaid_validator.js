#!/usr/bin/env node

/**
 * Mermaid Validation Script using basic syntax validation
 * Fallback approach when full mermaid.parse() has DOM issues in Node.js
 */

/**
 * Validates Mermaid script syntax using pattern matching
 * @param {string} script - The Mermaid script to validate
 * @returns {Promise<{valid: boolean, error: string|null}>}
 */
async function validateMermaidScript(script) {
    try {
        // Basic syntax validation patterns
        const lines = script.trim().split('\n');
        const errors = [];
        
        // Check for basic structure
        let hasGraphDeclaration = false;
        let subgraphStack = [];
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line || line.startsWith('%%')) continue;
            
            // Check for graph declaration
            if (/^graph\s+(TD|LR|TB|RL|BT)/i.test(line)) {
                hasGraphDeclaration = true;
                // Ensure uppercase direction
                const match = line.match(/graph\s+(\w+)/i);
                if (match && match[1] !== match[1].toUpperCase()) {
                    errors.push(`Line ${i + 1}: Direction '${match[1]}' must be UPPERCASE`);
                }
            }
            
            // Check for subgraph balance
            if (/^subgraph\s+/.test(line)) {
                const match = line.match(/subgraph\s+([A-Za-z_][A-Za-z0-9_]*)/);
                if (match) {
                    subgraphStack.push(match[1]);
                }
            } else if (/^\s*end\s*$/.test(line)) {
                if (subgraphStack.length === 0) {
                    errors.push(`Line ${i + 1}: Unmatched 'end' statement`);
                } else {
                    subgraphStack.pop();
                }
            }
            
            // Check for common syntax errors
            if (line.includes('&amp;amp;')) {
                errors.push(`Line ${i + 1}: Double-encoded ampersand '&amp;amp;' - use '&amp;' instead`);
            }
            
            if (line.includes('&') && !line.includes('&amp;') && line.includes('[')) {
                errors.push(`Line ${i + 1}: Unescaped ampersand - use '&amp;' in node labels`);
            }
            
            // Check for unquoted labels with special characters
            const nodeMatches = line.match(/\[[^\]]*\]/g);
            if (nodeMatches) {
                for (const nodeMatch of nodeMatches) {
                    if ((nodeMatch.includes('(') || nodeMatch.includes(')')) && 
                        !(nodeMatch.startsWith('["') && nodeMatch.endsWith('"]'))) {
                        errors.push(`Line ${i + 1}: Parentheses in labels must be quoted: ${nodeMatch}`);
                    }
                }
            }
        }
        
        // Final structure checks
        if (!hasGraphDeclaration) {
            errors.push("Missing graph declaration (e.g., 'graph TD')");
        }
        
        if (subgraphStack.length > 0) {
            errors.push(`Unbalanced subgraphs: missing 'end' for ${subgraphStack.join(', ')}`);
        }
        
        if (errors.length > 0) {
            return { 
                valid: false, 
                error: errors.join('; ')
            };
        }
        
        return { valid: true, error: null };
        
    } catch (error) {
        return { 
            valid: false, 
            error: `Validation system error: ${error.message}`
        };
    }
}

// CLI Interface - read script from command line argument
async function main() {
    const script = process.argv[2];
    
    if (!script) {
        console.error('Usage: node mermaid_validator.js "<mermaid_script>"');
        process.exit(1);
    }
    
    try {
        const result = await validateMermaidScript(script);
        
        // Output as JSON for Python to parse
        console.log(JSON.stringify(result));
        
        // Exit with appropriate code
        process.exit(result.valid ? 0 : 1);
        
    } catch (error) {
        console.error(JSON.stringify({
            valid: false,
            error: `Validation system error: ${error.message}`
        }));
        process.exit(1);
    }
}

// Only run if this script is executed directly
if (process.argv[1] === new URL(import.meta.url).pathname || process.argv[1].endsWith('mermaid_validator.js')) {
    main();
}

export { validateMermaidScript };