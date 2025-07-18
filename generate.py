# Re-import necessary library after code execution state reset
import pandas as pd

# Recreate the refined schedule
refined_schedule = [
    ("7月4日", "全天", "接机、入住、校园参观", "校外 → 力学楼", "组内学生陪同"),
    ("7月5日", "上午 9:00–11:00", "Lecture 1: Strategic Research Topic Selection", "力学楼", "全组"),
    ("7月5日", "下午 14:30–16:30", "交流讨论选题与可行性分析", "力学楼", "全组"),
    ("7月6日", "全天", "自由活动/休息", "长沙市区", "组内轮值陪同"),
    ("7月7日", "上午 9:00–11:00", "Lecture 2: AI for Literature Review", "力学楼", "全组"),
    ("7月7日", "下午 14:30–16:30", "申请书前期文献准备讨论", "力学楼", "小组讨论"),
    ("7月8日", "上午 9:00–11:00", "具体项目对接：晶体塑性建模接口规划", "力学楼", "项目组成员"),
    ("7月8日", "下午 14:30–16:30", "模型验证框架研讨", "力学楼", "项目组成员"),
    ("7月9日", "上午 9:00–11:00", "Lecture 3: Defining Paper Message", "力学楼", "全组"),
    ("7月9日", "下午 14:30–16:30", "论文写作核心内容研讨", "力学楼", "撰写成员"),
    ("7月10日", "上午 9:00–11:00", "赴长沙参加全国力学大会", "长沙", "卡利欧+组内2人"),
    ("7月10日", "下午 14:30–16:30", "大会专题报告与交流", "长沙", "卡利欧+组内2人"),
    ("7月11日", "上午 9:00–11:00", "参会专家交流反馈梳理", "长沙", "卡利欧+组内2人"),
    ("7月11日", "下午 14:30–16:30", "返程&总结会前阶段内容", "长沙→力学楼", "全组"),
    ("7月12日", "全天", "自由活动/休息", "本地景点", "组内轮值陪同"),
    ("7月13日", "上午 9:00–11:00", "Lecture 4: Clear Writing in Research", "力学楼", "全组"),
    ("7月13日", "下午 14:30–16:30", "图表设计与文献梳理实操", "力学楼", "全组"),
    ("7月14日", "上午 9:00–11:00", "项目组阶段成果汇报", "力学楼", "项目组成员"),
    ("7月14日", "下午 14:30–16:30", "撰写申请书部分草案内容", "力学楼", "写作成员"),
    ("7月15日", "上午 9:00–11:00", "Lecture 5: AI for Execution", "力学楼", "全组"),
    ("7月15日", "下午 14:30–16:30", "AI工具集成脚本演示", "力学楼", "技术人员参与"),
    ("7月16日", "上午 9:00–11:00", "外出交流：华中科技大学", "武汉华中科技大学", "卡利欧+组内2人"),
    ("7月16日", "下午 14:30–16:30", "项目方向交流与合作探讨", "武汉华中科技大学", "卡利欧+组内2人"),
    ("7月17日", "上午 9:00–11:00", "Lecture 6: Feedback & Impact", "力学楼", "全组"),
    ("7月17日", "下午 14:30–16:30", "模拟评审与意见回复训练", "力学楼", "全组"),
    ("7月18日", "上午 9:00–11:00", "工作进展反馈会", "力学楼", "全组"),
    ("7月18日", "下午 14:30–16:30", "前期申请书打磨与修改", "力学楼", "全组"),
    ("7月19日", "全天", "自由活动/休息", "城市周边", "组内轮值陪同"),
    ("7月20日", "上午 9:00–11:00", "Lecture 7: Funding & Collaboration", "力学楼", "全组"),
    ("7月20日", "下午 14:30–16:30", "国际合作申请实务分享", "力学楼", "有申报需求学生"),
    ("7月21日", "上午 9:00–11:00", "项目具体代码实现协助", "力学楼", "项目相关学生"),
    ("7月21日", "下午 14:30–16:30", "调试会议及问题答疑", "力学楼", "项目相关学生"),
    ("7月22日", "上午 9:00–11:00", "外出交流：昆明理工大学", "昆明理工大学", "卡利欧+组内2人"),
    ("7月22日", "下午 14:30–16:30", "联合课题框架研讨", "昆明理工大学", "卡利欧+组内2人"),
    ("7月23日", "上午 9:00–11:00", "Lecture: Advanced Crystal Modeling", "力学楼", "全组"),
    ("7月23日", "下午 14:30–16:30", "代码接口对接指导", "力学楼", "项目小组"),
    ("7月24日", "上午 9:00–11:00", "外出交流：江苏科技大学", "江苏科技大学", "卡利欧+组内2人"),
    ("7月24日", "下午 14:30–16:30", "后续课题合作与访问意向", "江苏科技大学", "卡利欧+组内2人"),
    ("7月25日", "上午 9:00–11:00", "最终项目成果展示", "力学楼", "全组"),
    ("7月25日", "下午 14:30–16:30", "经验交流与答疑", "力学楼", "全组"),
    ("7月26日", "全天", "自由活动/休息", "城市周边", "组内轮值陪同"),
    ("7月27日", "上午 9:00–11:00", "总结与反馈交流", "力学楼", "全组"),
    ("7月27日", "下午 14:30–16:30", "合影留念 & 自由提问", "力学楼", "全组"),
    ("7月28日", "上午", "返程准备", "宾馆", "—"),
    ("7月28日", "下午", "送站/送机", "校外", "组内2人陪同"),
]

# Save to Excel
refined_df = pd.DataFrame(refined_schedule, columns=["日期", "时间", "活动内容", "地点", "人员"])
refined_file_path = "C:/Users/Shuka/Desktop/卡利欧来华最终版日程安排.xlsx"
refined_df.to_excel(refined_file_path, index=False)

refined_file_path
