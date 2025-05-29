import json

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

PROMPT_TEMPLATE = """你是一位数据分析助手，你的回应内容取决于用户的请求内容，请按照下面的步骤处理用户请求：

1. 思考阶段 (Thought) ：先分析用户请求类型（文字回答/表格/图表），并验证数据类型是否匹配。
2. 行动阶段 (Action) ：根据分析结果选择以下严格对应的格式。
   - 纯文字回答: 
     {"answer": "不超过50个字符的明确答案"}

   - 表格数据：  
     {"table":{"columns":["列名1", "列名2", ...], "data":[["第一行值1", "值2", ...], ["第二行值1", "值2", ...]]}}

   - 柱状图 
     {"bar":{"columns": ["A", "B", "C", ...], "data":[35, 42, 29, ...]}}

   - 折线图 
     {"line":{"columns": ["A", "B", "C", ...], "data": [35, 42, 29, ...]}}

3. 格式校验要求
   - 字符串值必须使用英文双引号
   - 数值类型不得添加引号
   - 确保数组闭合无遗漏

   错误案例：{'columns':['Product', 'Sales'], data:[[A001, 200]]}  
   正确案例：{"columns":["product", "sales"], "data":[["A001", 200]]}

注意：响应数据的"output"中不要有换行符、制表符以及其他格式符号。

当前用户请求："""


# PROMPT_TEMPLATE = """
# 你是一位数据分析助手，你的回应内容取决于用户的请求内容。
#
# 1. 对于文字回答的问题，按照这样的格式回答：
#    {"answer": "<你的答案写在这里>"}
# 例如：
#    {"answer": "订单量最高的产品ID是'MNWC3-067'"}
#
# 2. 如果用户需要一个表格，按照这样的格式回答：
#    {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}
#
# 3. 如果用户的请求适合返回条形图，按照这样的格式回答：
#    {"bar": {"columns": ["A", "B", "C", ...], "data": [34, 21, 91, ...]}}
#
# 4. 如果用户的请求适合返回折线图，按照这样的格式回答：
#    {"line": {"columns": ["A", "B", "C", ...], "data": [34, 21, 91, ...]}}
#
# 5. 如果用户的请求适合返回散点图，按照这样的格式回答：
#    {"scatter": {"columns": ["A", "B", "C", ...], "data": [34, 21, 91, ...]}}
# 注意：我们只支持三种类型的图表："bar", "line" 和 "scatter"。
#
#
# 请将所有输出作为JSON字符串返回。请注意要将"columns"列表和数据列表中的所有字符串都用双引号包围。
# 例如：{"columns": ["Products", "Orders"], "data": [["32085Lip", 245], ["76439Eye", 178]]}
#
# 你要处理的用户请求如下： {input}
# """


def dataframe_agent(df, query):
    load_dotenv()
    model = ChatOpenAI(
        model="deepseek-chat",
        base_url='https://api.deepseek.com',
        api_key='sk-b09dda44fc5f40e9aeeda1f23b0ee51f',
        temperature=0
    )
    agent = create_pandas_dataframe_agent(
        llm=model,
        df=df,
        agent_executor_kwargs={"handle_parsing_errors": True},
        max_iterations=10,
        early_stopping_method='force',
        allow_dangerous_code=True,
        verbose=True
    )
    prompt = PROMPT_TEMPLATE + query
    response = agent.invoke({"input": prompt})
    return json.loads(response["output"])
