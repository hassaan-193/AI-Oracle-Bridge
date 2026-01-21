"""
Oracle MCP Server with Groq AI (Free Alternative)
Much faster and more generous rate limits than Gemini
"""

import sys
import re
import json
import asyncio

# ============================================================================
# DEPENDENCIES
# ============================================================================

def check_and_install_dependencies():
    """Check and install required packages."""
    required_packages = {
        'oracledb': 'oracledb',
        'mcp': 'mcp',
        'groq': 'groq',  # Groq instead of Gemini
        'nest_asyncio': 'nest_asyncio'
    }
    
    missing_packages = []
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("Installing missing packages...")
        import subprocess
        for package in missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
        print("✓ All packages installed")

check_and_install_dependencies()

import oracledb
import nest_asyncio
from groq import Groq
from mcp.server.fastmcp import FastMCP
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

nest_asyncio.apply()

# ============================================================================
# CONFIGURATION
# ============================================================================

DB_CONFIG = {
    'username': 'hr',
    'password': 'HR',
    'host': '172.21.110.133',
    'port': 1521,
    'service_name': 'FREEPDB1'
}

# Get your FREE Groq API key from: https://console.groq.com/keys
GROQ_API_KEY = "Enter your Secret API Key"

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def get_connection():
    """Create Oracle database connection."""
    try:
        dsn = f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['service_name']}"
        return oracledb.connect(
            user=DB_CONFIG['username'],
            password=DB_CONFIG['password'],
            dsn=dsn
        )
    except Exception as e:
        print(f"Database connection error: {e}")
        raise

def test_connection():
    """Test database connection."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT 'Connection successful' FROM dual")
        result = cur.fetchone()
        cur.close()
        conn.close()
        print(f"✓ {result[0]}")
        return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False

# ============================================================================
# SQL UTILITIES
# ============================================================================

FORBIDDEN_KEYWORDS = ["insert", "update", "delete", "drop", "alter", "truncate", "create", "merge", "grant", "revoke"]

def safe_sql(query: str) -> bool:
    """Check if query is safe (SELECT only)."""
    query_lower = query.lower().strip()
    if not query_lower.startswith("select"):
        return False
    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", query_lower):
            return False
    return True

def run_sql(query: str) -> dict:
    """Execute safe SELECT query."""
    if not safe_sql(query):
        return {"status": "blocked", "reason": "Only SELECT queries allowed"}
    
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description] if cur.description else []
        cur.close()
        conn.close()
        
        return {
            "status": "success",
            "columns": columns,
            "rows": rows,
            "row_count": len(rows)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ============================================================================
# MCP SERVER
# ============================================================================

mcp = FastMCP("oracle-mcp-server")

@mcp.tool()
def ping() -> str:
    """Test MCP server."""
    return "Oracle MCP Server is running!"

@mcp.tool()
def execute_sql(query: str) -> dict:
    """Execute SELECT query."""
    return run_sql(query)

@mcp.tool()
def list_tables() -> dict:
    """List all tables."""
    return run_sql("SELECT table_name FROM user_tables ORDER BY table_name")

@mcp.tool()
def describe_table(table_name: str) -> dict:
    """Get table structure."""
    query = f"""
        SELECT column_name, data_type, data_length, nullable
        FROM user_tab_columns
        WHERE table_name = UPPER('{table_name}')
        ORDER BY column_id
    """
    return run_sql(query)

@mcp.tool()
def get_system_info() -> dict:
    """Get database information."""
    query = """
        SELECT banner as version
        FROM v$version 
        WHERE banner LIKE 'Oracle%'
    """
    return run_sql(query)

# ============================================================================
# GROQ AI CLIENT
# ============================================================================

groq_client = Groq(api_key=GROQ_API_KEY)

async def chat_with_oracle(user_question: str):
    """Process question using Groq and Oracle MCP tools."""
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[__file__, "--server"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            tools_response = await session.list_tools()
            tools = tools_response.tools if hasattr(tools_response, 'tools') else []
            
            tool_descriptions = "\n".join([
                f"- {tool.name}: {tool.description}" 
                for tool in tools
            ])
            
            print(f"✓ Connected to MCP server with {len(tools)} tools\n")
            
            planning_prompt = f"""
You are an AI assistant with access to Oracle database tools.

Available tools:
{tool_descriptions}

User question: {user_question}

Respond in JSON format:
{{
    "tool_name": "name_of_tool",
    "parameters": {{"param1": "value1"}},
    "reasoning": "why you chose this tool"
}}

If no tool needed:
{{"tool_name": null, "direct_answer": "your answer"}}
"""
            
            print("🤔 Groq is analyzing your question...\n")
            
            try:
                # Use Groq API
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",  # Fast and smart model
                    messages=[{"role": "user", "content": planning_prompt}],
                    temperature=0.3,
                    max_tokens=1000
                )
                
                decision_text = response.choices[0].message.content.strip()
                if "```json" in decision_text:
                    decision_text = decision_text.split("```json")[1].split("```")[0]
                elif "```" in decision_text:
                    decision_text = decision_text.split("```")[1].split("```")[0]
                
                decision = json.loads(decision_text.strip())
                
                if decision.get("tool_name"):
                    tool_name = decision["tool_name"]
                    params = decision.get("parameters", {})
                    
                    print(f"🔧 Using tool: {tool_name}")
                    print(f"📝 Parameters: {json.dumps(params, indent=2)}\n")
                    
                    result = await session.call_tool(tool_name, params)
                    
                    if hasattr(result, 'content'):
                        tool_output = "\n".join([
                            item.text if hasattr(item, 'text') else str(item)
                            for item in result.content
                        ])
                    else:
                        tool_output = str(result)
                    
                    print(f"📊 Tool result received\n")
                    
                    formatting_prompt = f"""
The user asked: {user_question}

Tool '{tool_name}' returned:
{tool_output}

Provide a clear answer based on this data.
"""
                    
                    final_response = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": formatting_prompt}],
                        temperature=0.3,
                        max_tokens=1000
                    )
                    
                    print("=" * 60)
                    print("ANSWER:")
                    print("=" * 60)
                    print(final_response.choices[0].message.content)
                    print("=" * 60)
                
                else:
                    print("=" * 60)
                    print("DIRECT ANSWER:")
                    print("=" * 60)
                    print(decision.get("direct_answer", "No answer provided"))
                    print("=" * 60)
            
            except json.JSONDecodeError as e:
                print(f"Error parsing response: {e}")
            except Exception as e:
                print(f"Error: {e}")

async def interactive_mode():
    """Run interactive chat."""
    print("=" * 60)
    print("Oracle + Groq AI Interactive Chat")
    print("=" * 60)
    print("Ask questions about your Oracle database!")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        print()
        await chat_with_oracle(user_input)
        print()

# ============================================================================
# MAIN
# ============================================================================

def run_tests():
    """Run tests."""
    print("=" * 60)
    print("Running Setup Tests")
    print("=" * 60)
    
    print("\n1. Testing database connection...")
    db_ok = test_connection()
    
    print("\n2. Testing SQL safety...")
    sql_ok = True
    for query, expected in [("SELECT * FROM users", True), ("DROP TABLE users", False)]:
        result = safe_sql(query)
        print(f"{'✓' if result == expected else '✗'} Query: {query[:40]}... -> {result}")
        if result != expected:
            sql_ok = False
    
    print("\n3. Testing Groq API...")
    groq_ok = False
    try:
        if GROQ_API_KEY == "gsk_YOUR_GROQ_API_KEY_HERE":
            print("⚠️  Please set your GROQ_API_KEY in the script")
        else:
            test_response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "Say 'API works!'"}],
                max_tokens=10
            )
            print(f"✓ Groq API: {test_response.choices[0].message.content}")
            groq_ok = True
    except Exception as e:
        print(f"✗ Groq API error: {e}")
    
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    print(f"{'✓' if db_ok else '✗'} Database Connection")
    print(f"{'✓' if sql_ok else '✗'} SQL Safety")
    print(f"{'✓' if groq_ok else '✗'} Groq API")
    
    return db_ok and sql_ok and groq_ok

async def run_client():
    """Run AI client."""
    example_queries = [
        "List all tables in the database",
        "What is the database version?"
    ]
    
    print("\n" + "=" * 60)
    print("Running Example Queries")
    print("=" * 60)
    
    for i, query in enumerate(example_queries, 1):
        print(f"\n[{i}/{len(example_queries)}] Query: {query}")
        print("-" * 60)
        await chat_with_oracle(query)
        if i < len(example_queries):
            await asyncio.sleep(1)
    
    print("\n\nStarting interactive mode...\n")
    await interactive_mode()

def run_server():
    """Run MCP server."""
    print("Starting Oracle MCP Server...")
    mcp.run()

def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        run_server()
    elif len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_tests()
    else:
        print("=" * 60)
        print("Oracle MCP Server with Groq AI")
        print("=" * 60)
        print("\nGet FREE API key: https://console.groq.com/keys")
        print("=" * 60)
        
        if run_tests():
            print("\n🎉 All tests passed!")
            print("Starting AI client in 3 seconds...")
            import time
            time.sleep(3)
            try:
                asyncio.run(run_client())
            except KeyboardInterrupt:
                print("\n\nSession ended")
        else:
            print("\n⚠️ Fix issues above and try again")

if __name__ == "__main__":
    main()