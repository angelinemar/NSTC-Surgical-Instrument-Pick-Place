"""Small Tk helpers; no simulator imports or recording side effects."""
import tkinter as tk


class Tooltip:
    def __init__(self,widget,text):
        self.widget,self.text=widget,text
        self.job=None; self.popup=None
        widget.bind('<Enter>',self.schedule,add='+')
        widget.bind('<Leave>',self.hide,add='+')
        widget.bind('<ButtonPress>',self.hide,add='+')
        widget.bind('<Destroy>',self.hide,add='+')
    def schedule(self,event=None):
        self.hide(); self.job=self.widget.after(450,self.show)
    def show(self):
        self.job=None
        if not self.widget.winfo_exists(): return
        self.popup=tk.Toplevel(self.widget); self.popup.wm_overrideredirect(True)
        label=tk.Label(self.popup,text=self.text,wraplength=340,justify='left',padx=12,pady=9,
                       bg='#263b50',fg='#f2f7fc',relief='solid',borderwidth=1,font=('Segoe UI',10))
        label.pack(); self.popup.update_idletasks()
        x=min(self.widget.winfo_rootx(),self.widget.winfo_screenwidth()-self.popup.winfo_reqwidth()-10)
        y=self.widget.winfo_rooty()+self.widget.winfo_height()+5
        if y+self.popup.winfo_reqheight()>self.widget.winfo_screenheight():
            y=self.widget.winfo_rooty()-self.popup.winfo_reqheight()-5
        self.popup.geometry(f'+{max(0,x)}+{max(0,y)}')
    def hide(self,event=None):
        if self.job:
            self.widget.after_cancel(self.job); self.job=None
        if self.popup:
            self.popup.destroy(); self.popup=None


def cell_style(count,cycles,current=False):
    if current: return '#164d70','#64d6ff','CURRENT'
    if count>=cycles: return '#20573e','#74e5a5','DONE'
    if count: return '#24483f','#85bfa8','PARTIAL'
    return '#20343a','#577580','PENDING'
