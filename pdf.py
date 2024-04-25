import datetime

from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.shapes import Drawing, Circle
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Spacer, SimpleDocTemplate, Table, Paragraph, Image
from reportlab.graphics.shapes import Image as DrawingImage
import os
# 注册字体
song = "simsun"
pdfmetrics.registerFont(TTFont(song, "/home/www/project/hospital_2/hosptialapp/simsun.ttc"))

PAGE_HEIGHT = A4[1]
PAGE_WIDTH = A4[0]
#
# 设置段落格式
titleStyle = ParagraphStyle(
    name="titleStyle",
    alignment=1,
    fontName=song,
    fontSize=10,
    textColor=colors.black,
    # backColor=HexColor(0xF2EEE9),
    borderPadding=(5, 5)
)

def red_text(keywords,text):
    for item in keywords:
        if item in text:
            text='<font color="red">{}</font>'.format(text)
    return text
def DrawPageInfo(c: Canvas, date=datetime.date.today()):
    """绘制页脚"""
    # 设置边框颜色
    c.setStrokeColor(colors.dimgrey)
    # 绘制线条
    c.line(70, PAGE_HEIGHT - 790, 520, PAGE_HEIGHT - 790)
    # 绘制页脚文字
    c.setFont(song, 8)
    c.setFillColor(colors.black)
    # c.drawString(70, PAGE_HEIGHT - 780, '报告医生:XXX')
    # c.drawString(460,PAGE_HEIGHT - 780, '审核医生：XXX')
    c.drawString(70, PAGE_HEIGHT - 805, f"生成日期：{date.isoformat()}")


# 绘制用户信息表
def drawUserInfoTable(c: Canvas, x, y, name, gender, age, time,result,remark,name_docter,keywords):
    """绘制用户信息表"""
    # data = [["姓名", "龙在天"],
    #         ["性别", '男'],
    #         ["年龄", "12"],
    #         ["时间", "2000-01-01"]]
    c.line(70, PAGE_HEIGHT - 140, 520, PAGE_HEIGHT - 140)
    data = [["姓名:", name,"性别:", gender],["年龄:", age,"时间:", time]]
    c.line(70, PAGE_HEIGHT - 190, 520, PAGE_HEIGHT - 190)
    # c.line(30, 10, 570, 10)
    # t=Table(data)
    t = Table(data, style={
        ("FONT", (0, 0), (-1, -1), song, 10),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
        ('ALIGN', (-1, -1), (-1, -1), 'CENTER')
    },
        colWidths=[1*cm,2*cm,1*cm,2*cm]
          # colWidths=(500/8)*inch
              )
    t._argW[1] = 200
    t.wrapOn(c, 0, 0)
    t.drawOn(c, x, y)
    c.setFont(song, 15)
    c.drawString(80, PAGE_HEIGHT - 220, '结果：')
    c.setFont(song, 10)
    Paragraph(result, titleStyle)
    # t1=c.beginText()
    # t1.setFont(song,10)
    # t1.setTextOrigin(80,PAGE_HEIGHT-240)
    # t1.textLine(result)
    # c.drawText(t1)
    c.setFont(song,10)
    c.setFillColor(colors.black)
    c.drawString(80, PAGE_HEIGHT - 240, result)
    c.setFont(song, 15)
    c.setFillColor(colors.orange)
    c.drawString(80, PAGE_HEIGHT - 400, '备注：')
    c.setFont(song, 10)
    c.setFillColor(colors.black)
    x=80
    y=PAGE_HEIGHT - 420
    words=list(remark)
    i=0
    while i < len(words):
        flag = 1
        loc = -1
        c.setFillColor(colors.black)
        for j in range(0,len(keywords)):
            flag = 1
            loc=-1
            if words[i]==keywords[j][0]:
                k=0
                while k <len(keywords[j]):
                    if words[i+k]!=keywords[j][k]:
                        flag=1
                        loc=-1
                        k+=1
                        break
                    else:
                        k+=1
            else:
                continue
            if k == len(keywords[j]):
                flag = 0
                loc = j
                break
        if flag == 0:
            c.setFillColor(colors.red)
            m=0
            while m < len(keywords[loc]):
                c.drawString(x, y, words[i] + "")
                x += c.stringWidth(words[i] + "")
                i+=1
                m+=1
        else:
            c.drawString(x, y, words[i] + "")
            x += c.stringWidth(words[i] + "")
            i+=1


    # c.drawString(80, PAGE_HEIGHT - 450, remark)
    c.setFont(song, 8)
    c.setFillColor(colors.black)
    c.drawString(70, PAGE_HEIGHT - 780, f"报告医生：{name_docter}")
    c.drawString(460, PAGE_HEIGHT - 780, f"审核医生：{name_docter}")



# 绘制饼图
def drawScorePie(canvas: Canvas, x, y, score):
    """绘制评分饼图"""
    d = Drawing(100, 100)
    # 画大饼图
    pc = Pie()
    pc.width = 100
    pc.height = 100
    # 设置数据
    pc.data = [score, 100 - score]
    pc.slices.strokeWidth = 0.1
    # 设置颜色
    color = colors.seagreen
    if (score < 60):
        color = colors.orangered
    elif (score < 85):
        color = colors.orange
    pc.slices.strokeColor = colors.transparent
    pc.slices[0].fillColor = color
    pc.slices[1].fillColor = colors.transparent
    d.add(pc)
    # 画内圈
    circle = Circle(50, 50, 40)
    circle.fillColor = colors.white
    circle.strokeColor = colors.transparent
    d.add(circle)
    # 把饼图画到Canvas上
    d.drawOn(canvas, x, y)
    # 写字
    canvas.setFont(song, 30)
    canvas.setFillColor(color)
    canvas.drawCentredString(x + 50, y + 40, f"2023")


def myFirstPage(c: Canvas, doc):
    c.saveState()
    # 设置填充色
    c.setFillColor(colors.orange)
    # 设置字体大小
    c.setFont(song, 30)
    # 绘制居中标题文本
    c.drawCentredString(300, PAGE_HEIGHT - 80, "武汉大学 B426医院")
    c.drawCentredString(300,PAGE_HEIGHT - 120,'病历报告单')
    # 绘制表格
    # drawUserInfoTable(c, 80, PAGE_HEIGHT - 180,'Mary','女','12','2021-12-1','肝炎','pdf','Linda')

    drawUserInfoTable(c,80,PAGE_HEIGHT - 180,doc.name,doc.gender,doc.age,doc.time,doc.result,doc.remark,doc.name_docter,["疾病",'的'])
    # 绘制饼图
    # drawScorePie(c, 360, PAGE_HEIGHT - 200, 70)
    # 绘制页脚
    DrawPageInfo(c)
    c.restoreState()


def myLaterPages(c: Canvas, doc):
    c.saveState()
    # 绘制页脚
    DrawPageInfo(c)
    c.restoreState()

#
# 创建文档
doc = SimpleDocTemplate("/home/www/project/hospital_2/hosptialapp/output/pdftest.pdf")
Story = [Spacer(1, 2 * inch)]
# 绘制段落
# Story.append(Paragraph("我是龙在天涯，在这里祝大家：", titleStyle))
# Story.append(Spacer(1, 0.2 * inch))
# Story.append(Paragraph("新年快乐", titleStyle))
# Story.append(Spacer(1, 0.2 * inch))
# Story.append(Paragraph("恭喜发财", titleStyle))
# Story.append(Spacer(1, 0.2 * inch))
# Story.append(Paragraph("虎年大吉", titleStyle))
# Story.append(Spacer(1, 0.2 * inch))
# 绘制顺序排列的图像
# Story.append(Image("data/img1.png", 6 * inch, 3 * inch))
# 绘制两个横向排布的图像
# d = Drawing()
# d.add(DrawingImage(0, 0, 200, 200, "data/img2.png"))
# d.add(DrawingImage(200, 0, 200, 200, "data/img3.png"))
# Story.append(d)
# 保存文档
# name, gender, age, time,result,remark,name_docter
doc.name='Mary'
doc.gender='女'
doc.age='18'
doc.time=date=datetime.date.today().isoformat()
doc.result='肺炎'
doc.remark="这是一段有关病症症状的文本，疾病的症状很多，其中有一项症状比较常见"
# doc.remark='the key word remark to red'
doc.name_docter='Linda'
doc.build(Story, onFirstPage=myFirstPage, onLaterPages=myLaterPages)
