\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tmlr}[2025/03/01 Minimal TMLR class]
\LoadClass[11pt]{article}
\RequirePackage{geometry}
\geometry{margin=1in}
\RequirePackage{setspace}
\setstretch{1.05}
\setlength{\parskip}{6pt}
\setlength{\parindent}{0pt}
\makeatletter
\newcommand{\tmlrvolume}[1]{\gdef\@tmlrvolume{#1}}
\newcommand{\tmlryear}[1]{\gdef\@tmlryear{#1}}
\newcommand{\tmlrsubmissionnumber}[1]{\gdef\@tmlrsubmissionnumber{#1}}
\newcommand{\tmlrproceeding}{\@ifstar{\gdef\@tmlrproceeding{}}{\gdef\@tmlrproceeding{}}}
\newcommand{\tmlrfinalcopy}[1]{\gdef\@tmlrfinalcopy{#1}}
\newcommand{\tmrllogourl}[1]{\gdef\@tmrllogourl{#1}}
\newcommand{\editor}[1]{\gdef\@tmrleditor{#1}}
\newcommand{\reviewer}[1]{\gdef\@tmlrreviewer{#1}}
\newcommand{\tmrldoi}[1]{\gdef\@tmrldoi{#1}}
\newcommand{\tmrlyear}[1]{\gdef\@tmlryear{#1}}
\newcommand{\keywords}[1]{\gdef\@tmrlkeywords{#1}}
\renewcommand{\maketitle}{%
  \begin{center}%
    {\LARGE\bfseries \@title\par}\vspace{1em}%
    {\large \@author\par}\vspace{0.5em}%
    {\normalsize \@date\par}%
  \end{center}%
  \vspace{1em}%
}
\makeatother
