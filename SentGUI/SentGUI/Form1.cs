using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace SentGUI
{
    public partial class Form1 : Form
    {
        private Process cmd;

        public Form1()
        {
            InitializeComponent();
            cmd = new Process
            {
                StartInfo =
                {
                    FileName = "python.exe",
                    Arguments = "SentimentAnalyzer.py",
                    RedirectStandardInput = true,
                    RedirectStandardOutput = true,
                    CreateNoWindow = true,
                    UseShellExecute = false
                }
            };
            cmd.Start();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            var txt = richTextBox1.Text.Replace("\n", " ").Replace("\r", "");
            txt = Encoding.ASCII.GetString(Encoding.Convert(Encoding.UTF8, Encoding.ASCII, Encoding.UTF8.GetBytes(txt)));
            cmd.StandardInput.WriteLine(txt);
            cmd.StandardInput.Flush();
            var result = cmd.StandardOutput.ReadLine();
            label2.Text = result;
        }
    }
}
