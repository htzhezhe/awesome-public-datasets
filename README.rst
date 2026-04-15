# AI Agent 完全教程：从理论到实践，从单体到多体

---

## 第一部分：理论基础

### 1.1 什么是 Agent？

Agent（智能体）是一个能够**感知环境、自主决策、执行行动**的系统。与传统的"一问一答"式 LLM 调用不同，Agent 具备以下核心能力：

- **自主规划**：将复杂任务拆解为可执行的子步骤
- **工具调用**：通过 API、代码执行等方式与外部世界交互
- **记忆管理**：在多轮交互中维持上下文和状态
- **反思纠错**：评估执行结果，必要时调整策略

一个直观的类比：传统 LLM 像一个"顾问"——你问它问题，它给你建议；而 Agent 像一个"员工"——你给它目标，它自己想办法完成。

### 1.2 Agent 的核心循环

几乎所有 Agent 框架都遵循同一个循环模式，称为 **ReAct 循环**（Reasoning + Acting）：

```
┌─────────────────────────────────────────┐
│                                         │
│   感知（Observe）                        │
│     ↓                                   │
│   思考（Think / Reason）                 │
│     ↓                                   │
│   行动（Act / Tool Call）                │
│     ↓                                   │
│   观察结果（Observe Result）             │
│     ↓                                   │
│   是否完成？── 否 ──→ 回到"思考"         │
│     │                                   │
│     是                                  │
│     ↓                                   │
│   输出最终结果                           │
│                                         │
└─────────────────────────────────────────┘
```

### 1.3 Agent 的四大核心模块

| 模块 | 作用 | 举例 |
|------|------|------|
| **大脑（LLM）** | 推理、规划、决策 | Claude, GPT-4, Gemini |
| **工具（Tools）** | 与外部世界交互 | 搜索引擎、代码解释器、API |
| **记忆（Memory）** | 维持状态与上下文 | 短期记忆（对话历史）、长期记忆（向量数据库） |
| **规划（Planning）** | 任务分解与执行策略 | Chain-of-Thought、Tree-of-Thought |

---

## 第二部分：单 Agent 实战

### 2.1 最简 Agent：从零搭建

下面用 Python 伪代码展示一个最小可运行的 Agent：

```python
import anthropic

client = anthropic.Anthropic()

# 第一步：定义工具
tools = [
    {
        "name": "search_web",
        "description": "搜索互联网获取最新信息",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "run_python",
        "description": "执行 Python 代码并返回结果",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "要执行的 Python 代码"}
            },
            "required": ["code"]
        }
    }
]

# 第二步：实现工具执行器
def execute_tool(tool_name, tool_input):
    if tool_name == "search_web":
        return do_web_search(tool_input["query"])
    elif tool_name == "run_python":
        return run_code(tool_input["code"])

# 第三步：Agent 主循环
def run_agent(user_task):
    messages = [{"role": "user", "content": user_task}]

    while True:
        # 调用 LLM
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system="你是一个能使用工具完成任务的智能助手。仔细思考后再行动。",
            tools=tools,
            messages=messages,
        )

        # 检查是否需要调用工具
        if response.stop_reason == "tool_use":
            # 提取工具调用
            tool_block = next(b for b in response.content if b.type == "tool_use")

            # 执行工具
            result = execute_tool(tool_block.name, tool_block.input)

            # 将结果追加到消息历史
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": str(result)
                }]
            })
        else:
            # 任务完成，返回最终结果
            final_text = next(b.text for b in response.content if b.type == "text")
            return final_text
```

### 2.2 关键设计原则

**原则一：工具描述要精确**

工具的 `description` 是 Agent 决策的关键依据。模糊的描述会导致错误的工具选择。

```python
# ❌ 差的描述
{"name": "calc", "description": "做计算"}

# ✅ 好的描述
{"name": "calc", "description": "对数学表达式求值。输入标准数学表达式如 '2+3*4'，返回计算结果。适用于算术运算，不适用于符号推导。"}
```

**原则二：System Prompt 定义人格与边界**

```python
system_prompt = """
你是一个数据分析助手。你的工作流程：
1. 先理解用户需求，明确分析目标
2. 使用 run_python 工具加载和探索数据
3. 执行分析并生成可视化
4. 用通俗语言总结发现

约束：
- 不要一次执行过多代码，分步进行
- 每次执行后检查结果再继续
- 如果数据有问题，主动告知用户
"""
```

**原则三：错误处理与重试**

```python
def execute_tool_safely(tool_name, tool_input, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = execute_tool(tool_name, tool_input)
            return {"status": "success", "result": result}
        except Exception as e:
            if attempt == max_retries - 1:
                return {"status": "error", "error": str(e)}
            # 将错误信息反馈给 Agent，让它调整策略
```

### 2.3 单 Agent 常用模式

#### 模式 A：ReAct（推理 + 行动）

最经典的模式，Agent 在每次行动前先"思考"。

```
思考：用户想知道今天的天气，我需要调用搜索工具
行动：search_web("温哥华今日天气")
观察：晴，15°C，湿度 60%
思考：我已经获得了天气信息，可以回答了
最终回答：今天温哥华天气晴朗，气温 15°C...
```

#### 模式 B：Plan-then-Execute（先规划后执行）

适合复杂任务，先生成完整计划，再逐步执行。

```python
def plan_and_execute(task):
    # 阶段一：生成计划
    plan = call_llm(f"请为以下任务制定详细执行计划：{task}")

    # 阶段二：逐步执行
    results = []
    for step in plan.steps:
        result = execute_step(step)
        results.append(result)

        # 可选：根据结果动态调整后续计划
        if needs_replan(result):
            plan = replan(plan, results)

    return synthesize(results)
```

#### 模式 C：Reflection（自我反思）

Agent 执行后回头检查自己的输出质量。

```python
def agent_with_reflection(task):
    # 第一次尝试
    draft = run_agent(task)

    # 自我反思
    critique = call_llm(f"""
        任务：{task}
        你的输出：{draft}

        请严格评估这个输出：
        1. 是否完整回答了任务？
        2. 有没有事实错误？
        3. 有什么可以改进的地方？
    """)

    # 根据反思改进
    if needs_improvement(critique):
        improved = call_llm(f"根据以下反馈改进你的回答：{critique}")
        return improved

    return draft
```

---

## 第三部分：Multi-Agent 系统

### 3.1 为什么需要多 Agent？

单 Agent 的局限性：

- **上下文窗口有限**：复杂任务会撑爆上下文
- **专业性不足**：一个 Agent 难以精通所有领域
- **可靠性风险**：单点失败会导致整个任务失败
- **并发效率**：串行执行效率低下

多 Agent 的优势在于**分工协作**，就像一个团队比一个全能选手更强大。

### 3.2 Multi-Agent 的四大架构模式

#### 模式一：Orchestrator（编排者模式）

一个"经理" Agent 负责分配任务，多个"员工" Agent 负责执行。

```
              ┌─────────────┐
              │  Orchestrator│
              │  (编排者)     │
              └──────┬──────┘
           ┌─────────┼─────────┐
           ▼         ▼         ▼
      ┌────────┐ ┌────────┐ ┌────────┐
      │ Agent A│ │ Agent B│ │ Agent C│
      │ (研究) │ │ (编码) │ │ (审查) │
      └────────┘ └────────┘ └────────┘
```

```python
class OrchestratorAgent:
    def __init__(self):
        self.workers = {
            "researcher": ResearchAgent(),
            "coder": CodingAgent(),
            "reviewer": ReviewAgent(),
        }

    def run(self, task):
        # 分析任务，制定计划
        plan = self.plan(task)

        results = {}
        for step in plan:
            worker = self.workers[step.assigned_to]
            result = worker.execute(step.instruction)
            results[step.id] = result

            # 编排者检查中间结果，决定下一步
            if not self.is_satisfactory(result):
                result = self.reassign_or_retry(step, result)

        return self.synthesize(results)
```

#### 模式二：Pipeline（流水线模式）

Agent 按固定顺序串行处理，每个 Agent 的输出是下一个的输入。

```
  输入 → [Agent A: 研究] → [Agent B: 写作] → [Agent C: 审校] → 输出
```

```python
class Pipeline:
    def __init__(self, agents: list):
        self.agents = agents

    def run(self, initial_input):
        current = initial_input
        for agent in self.agents:
            current = agent.process(current)
        return current

# 使用示例：内容创作流水线
pipeline = Pipeline([
    ResearchAgent(system="你是研究员，负责收集资料"),
    WriterAgent(system="你是作家，根据资料撰写文章"),
    EditorAgent(system="你是编辑，负责润色和纠错"),
])

result = pipeline.run("写一篇关于量子计算的科普文章")
```

#### 模式三：Debate / 辩论模式

多个 Agent 对同一问题给出不同观点，通过辩论提高结论质量。

```python
class DebateSystem:
    def __init__(self):
        self.debaters = [
            Agent(system="你是乐观派分析师，关注机遇和积极面"),
            Agent(system="你是谨慎派分析师，关注风险和潜在问题"),
        ]
        self.judge = Agent(system="你是裁判，综合各方观点给出平衡结论")

    def run(self, question, rounds=2):
        arguments = []

        for round_num in range(rounds):
            for debater in self.debaters:
                context = format_previous_arguments(arguments)
                arg = debater.respond(f"{question}\n\n之前的讨论：\n{context}")
                arguments.append(arg)

        # 裁判综合判断
        verdict = self.judge.respond(
            f"问题：{question}\n辩论记录：\n{format_all(arguments)}\n请给出综合结论。"
        )
        return verdict
```

#### 模式四：Autonomous Swarm（自主集群）

Agent 之间没有固定层级，根据需要动态协作。每个 Agent 可以自主决定是否"呼叫"其他 Agent。

```python
class SwarmAgent:
    def __init__(self, name, specialty, registry):
        self.name = name
        self.specialty = specialty
        self.registry = registry  # 共享的 Agent 注册表

    def process(self, task):
        # 尝试自己处理
        result = self.attempt(task)

        if self.needs_help(result):
            # 查找合适的协作者
            helper = self.registry.find_agent(needed_skill=result.gap)
            sub_result = helper.process(result.sub_task)
            result = self.incorporate(result, sub_result)

        return result

# 注册多个 Agent
registry = AgentRegistry()
registry.register(SwarmAgent("数据专家", "数据分析与可视化", registry))
registry.register(SwarmAgent("文案专家", "文案撰写与润色", registry))
registry.register(SwarmAgent("代码专家", "编写与调试代码", registry))
```

### 3.3 Agent 间通信设计

Agent 之间需要传递结构化的信息。一个实用的消息格式：

```python
@dataclass
class AgentMessage:
    sender: str           # 发送者 Agent 名称
    receiver: str         # 接收者 Agent 名称
    msg_type: str         # "task" | "result" | "question" | "feedback"
    content: str          # 消息正文
    context: dict         # 附加上下文
    priority: int         # 优先级

# 示例
msg = AgentMessage(
    sender="orchestrator",
    receiver="researcher",
    msg_type="task",
    content="查找 2025 年全球 AI 芯片市场规模数据",
    context={"deadline": "2min", "format": "structured_data"},
    priority=1
)
```

### 3.4 共享记忆与状态管理

多 Agent 系统中，共享状态是关键挑战：

```python
class SharedMemory:
    """多 Agent 共享的状态存储"""

    def __init__(self):
        self.facts = {}        # 确认的事实
        self.tasks = {}        # 任务状态追踪
        self.artifacts = {}    # 产出物（文件、代码等）
        self.conversation = [] # 交互历史

    def add_fact(self, agent_name, key, value, confidence=1.0):
        self.facts[key] = {
            "value": value,
            "source": agent_name,
            "confidence": confidence,
            "timestamp": now()
        }

    def get_context_for(self, agent_name):
        """为特定 Agent 生成相关上下文摘要"""
        relevant = filter_by_relevance(self.facts, agent_name)
        return format_context(relevant)
```

---

## 第四部分：实战案例

### 案例一：自动化研究报告生成（Multi-Agent Pipeline）

```python
# 三个专业 Agent 组成流水线
agents = {
    "researcher": Agent(
        system="""你是研究员。
        1. 根据主题搜索 5-10 篇权威资料
        2. 提取关键数据和观点
        3. 输出结构化的研究摘要""",
        tools=[search_web, fetch_url]
    ),

    "analyst": Agent(
        system="""你是数据分析师。
        1. 接收研究员的原始资料
        2. 识别趋势和模式
        3. 生成数据可视化建议
        4. 输出分析报告""",
        tools=[run_python, create_chart]
    ),

    "writer": Agent(
        system="""你是专业撰稿人。
        1. 根据研究和分析结果撰写报告
        2. 确保逻辑清晰、语言流畅
        3. 包含引用来源
        4. 输出完整的 Markdown 报告""",
        tools=[write_file]
    ),
}

def generate_report(topic):
    research = agents["researcher"].run(f"深入研究：{topic}")
    analysis = agents["analyst"].run(f"分析以下资料：\n{research}")
    report = agents["writer"].run(
        f"根据以下研究和分析撰写报告：\n研究：{research}\n分析：{analysis}"
    )
    return report
```

### 案例二：代码开发团队（Orchestrator 模式）

```python
class DevTeam:
    def __init__(self):
        self.pm = Agent(system="你是项目经理，负责需求分析和任务拆解")
        self.dev = Agent(system="你是开发工程师，负责编写高质量代码",
                        tools=[write_code, run_tests])
        self.qa = Agent(system="你是 QA 工程师，负责代码审查和测试",
                       tools=[read_code, run_tests])

    def build_feature(self, requirement):
        # PM 拆解需求
        tasks = self.pm.run(f"拆解需求为开发任务：\n{requirement}")

        for task in tasks:
            # 开发者编码
            code = self.dev.run(f"实现以下功能：\n{task}")

            # QA 审查
            review = self.qa.run(f"审查以下代码：\n{code}")

            # 如果审查不通过，重新开发
            while not review.approved:
                code = self.dev.run(f"根据反馈修改代码：\n{review.feedback}")
                review = self.qa.run(f"重新审查：\n{code}")

        return "Feature complete"
```

---

## 第五部分：最佳实践与常见陷阱

### ✅ 最佳实践

1. **从简单开始**：先用单 Agent 验证核心流程，确认工具可靠后再扩展到多 Agent
2. **明确职责边界**：每个 Agent 的 system prompt 要清楚定义它的角色、能力和限制
3. **结构化通信**：Agent 之间传递 JSON 而非自由文本，减少信息丢失
4. **设置安全阀**：限制最大循环次数、最大 token 消耗、超时时间
5. **可观测性**：记录每个 Agent 的每次调用、输入输出，方便调试
6. **人类在回路（Human-in-the-Loop）**：关键决策点让人类确认

### ❌ 常见陷阱

1. **无限循环**：Agent 反复调用工具却不收敛 → 设置 `max_iterations`
2. **上下文爆炸**：对话历史太长 → 定期摘要压缩
3. **过度设计**：不是所有任务都需要多 Agent → 简单任务用单 Agent
4. **幻觉传播**：一个 Agent 的错误输出被下游 Agent 当作事实 → 加入验证环节
5. **协调开销**：Agent 之间沟通成本过高 → 减少不必要的交互轮次

### 选择建议

| 场景 | 推荐方案 |
|------|---------|
| 简单的工具调用任务 | 单 Agent + ReAct |
| 多步骤线性流程 | Pipeline（流水线） |
| 复杂项目管理 | Orchestrator（编排者） |
| 需要多角度分析 | Debate（辩论） |
| 高度动态的任务 | Swarm（集群） |

---

## 第六部分：推荐工具与框架

| 框架 | 特点 | 适用场景 |
|------|------|---------|
| **Claude API + Tool Use** | 原生工具调用，简洁高效 | 生产级单 Agent |
| **LangGraph** | 基于图的 Agent 编排 | 复杂多 Agent 工作流 |
| **CrewAI** | 角色扮演式多 Agent | 快速原型、团队协作模拟 |
| **AutoGen** | 微软出品，对话式多 Agent | 研究探索、代码生成 |
| **Swarm (OpenAI)** | 轻量级 Agent 切换 | 客服、路由类场景 |

---

> **总结**：Agent 的本质是"LLM + 循环 + 工具"。从一个简单的 ReAct 循环开始，逐步添加工具、记忆、反思能力，最后根据任务复杂度决定是否引入多 Agent 协作。记住——最好的架构是**最简单的能完成任务的架构**。
