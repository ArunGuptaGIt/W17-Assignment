import ast
import operator as op
import platform
import sys
import psutil
from typing import Any, Dict, List

# Safe AST Math Evaluator
_SAFE_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.Mod: op.mod
}

def safe_eval_expr(expr: str) -> float:
    """Evaluate mathematical expression safely using AST parsing."""
    def _eval(node):
        if isinstance(node, ast.Num):  # <3.8
            return node.n
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            return _SAFE_OPERATORS[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            return _SAFE_OPERATORS[type(node.op)](operand)
        else:
            raise ValueError(f"Unsupported mathematical expression element: {ast.dump(node)}")

    parsed = ast.parse(expr.strip(), mode='eval')
    return float(_eval(parsed.body))

# Function executors
def execute_calculator(expression: str) -> Dict[str, Any]:
    try:
        result = safe_eval_expr(expression)
        return {"expression": expression, "result": result, "status": "success"}
    except Exception as e:
        return {"expression": expression, "error": str(e), "status": "error"}

def execute_system_info() -> Dict[str, Any]:
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    return {
        "os": f"{platform.system()} {platform.release()}",
        "python_version": sys.version.split()[0],
        "cpu_usage_percent": cpu_percent,
        "ram_total_gb": round(memory.total / (1024 ** 3), 2),
        "ram_available_gb": round(memory.available / (1024 ** 3), 2),
        "ram_used_percent": memory.percent,
        "status": "success"
    }

from typing import Any, Callable, Dict, List, Optional

# Tool JSON Schema Definitions for OpenAI/LiteLLM Function Calling API
TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Retrieve relevant document chunks using dense vector search, BM25 sparse search, or hybrid RRF fusion with cross-encoder reranking. Evaluates relevance and surfaces retrieval scores.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query or keyword phrase to find relevant context documents."
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["dense", "bm25", "hybrid"],
                        "default": "hybrid",
                        "description": "Retrieval strategy mode ('dense' for semantic search, 'bm25' for keyword search, 'hybrid' for combined search)."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform mathematical calculations safely (e.g. '150 * 0.2', '2**10', '(50 + 20) / 5').",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression string to evaluate."
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "system_info",
            "description": "Retrieve current system hardware and OS metrics (CPU usage, available RAM, OS info).",
            "parameters": {
                "type": "object",
                "properties": {
                    "detail_level": {
                        "type": "string",
                        "enum": ["summary", "full"],
                        "description": "Detail level of system metrics to retrieve.",
                        "default": "summary"
                    }
                },
                "required": []
            }
        }
    }
]

def dispatch_tool_call(
    tool_name: str,
    arguments: Dict[str, Any],
    search_handler: Optional[Callable[..., Any]] = None
) -> Dict[str, Any]:
    """Execute tool by name and return result dictionary."""
    valid_tools = [t["function"]["name"] for t in TOOLS_SCHEMA]
    if tool_name == "search_documents":
        query = arguments.get("query", "")
        mode = arguments.get("mode", "hybrid")
        if search_handler:
            try:
                return search_handler(query=query, mode=mode)
            except Exception as e:
                return {"error": f"Error executing search_documents: {str(e)}", "status": "error"}
        else:
            return {"error": "search_documents handler not registered.", "status": "error"}
    elif tool_name == "calculator":
        return execute_calculator(expression=arguments.get("expression", ""))
    elif tool_name == "system_info":
        return execute_system_info()
    else:
        return {
            "error": f"Tool '{tool_name}' does not exist. Available tools: {', '.join(valid_tools)}. Answer the user prompt directly.",
            "status": "error"
        }
