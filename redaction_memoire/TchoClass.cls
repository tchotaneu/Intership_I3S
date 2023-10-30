%%%% classe principale %%%%
	
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{TchoClass}
\LoadClass[a4paper,11pt]{report}

%%%%Structure%%%%

\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
%Language
\RequirePackage{csquotes}
\RequirePackage[english]{babel}
\RequirePackage[backend=biber,citestyle=alphabetic,bibstyle=authoryear]{biblatex}

\RequirePackage{tabularx}
\RequirePackage[top=2cm,bottom=2cm,left=2cm,right=2cm]{geometry}
\RequirePackage{hyperref}
\RequirePackage{enumitem}
\RequirePackage{xkeyval}
\RequirePackage{ifthen}

%%%%package tikz%%%%

\RequirePackage{tikz}
\usetikzlibrary{mindmap}
\RequirePackage{chemfig}
\RequirePackage{pgfplots}
\pgfplotsset{compat=newest}

%%%%Forme%%%%

%Chapter form
\RequirePackage[Bjornstrup]{fncychap}
\usepackage{everypage}
%package pour image 
\RequirePackage{graphicx}
\RequirePackage{textcomp}
\RequirePackage{eurosym}
\RequirePackage{mwe}
\RequirePackage{listings}
\RequirePackage{etoolbox}
\RequirePackage{setspace}
\RequirePackage{ccaption}
\RequirePackage{colortbl}
\RequirePackage{framed}
\RequirePackage{xcolor}
\RequirePackage{fourier}
\RequirePackage[pages=some]{background}
\setlength{\parindent}{2em}


%%%%pour les formulles mathematique %%%%

\RequirePackage{mathtools}
\RequirePackage{amsmath}
\RequirePackage{amsfonts}
\RequirePackage{amssymb}
\RequirePackage{mathrsfs}
\usetikzlibrary{arrows,matrix,positioning}
\usetikzlibrary{3d,calc}
\RequirePackage[makeroom]{cancel}
\RequirePackage{esint}

%%%%Utitile pour Annexe%%%%

\RequirePackage[toc,page]{appendix}
\linespread{1.5}
\renewcommand{\appendixtocname}{Annexes}
\renewcommand{\appendixpagename}{Annexes}

%%%%les Couleurs%%%%

\definecolor{apricot}{rgb}{0.98, 0.81, 0.69}
\definecolor{bleu}{rgb}{0.19, 0.55, 0.91}
\definecolor{rouge}{rgb}{0.77, 0.12, 0.23}
\definecolor{vert}{rgb}{0.13, 0.55, 0.13}
\definecolor{antiquefuchsia}{rgb}{0.57, 0.36, 0.51}
\colorlet{shadecolor}{cyan!30!white}
\colorlet{framecolor}{blue!50!black}

%%%%Custom command%%%%

\newcommand{\rouge}[1]{\textcolor{rouge}{#1}}
\newcommand{\bleu}[1]{\textcolor{bleu}{#1}}
\newcommand{\verts}[1]{\textcolor{vert}{#1}}

\makeatletter

%definition des varaibles ou keys 
\define@cmdkey [PRE] {fam} {title}{}
\define@cmdkey [PRE] {fam} {studies}{}
\define@cmdkey [PRE] {fam} {course}{}
\define@cmdkey [PRE] {fam} {author}{}
\define@cmdkey [PRE] {fam} {teacher}{}
\define@cmdkey [PRE] {fam} {date}{}
\define@cmdkey [PRE] {fam} {logo}{}
\define@cmdkey [PRE] {fam} {memoiretitle}{} 
\define@cmdkey [PRE] {fam} {specialite}{} 
\define@cmdkey [PRE] {fam} {manyAuthor}[1]{}
%les valeur par defaut des cles pour pouvoir  effectuer des test de condition pour remplir les %champs 
\presetkeys         [PRE] {fam} {title=titre,
								 studies=,
								 course=,
								 author=Auteurs,
								 teacher=,
								 date=,
								 logo=7,
                                 manyAuthor=8,}{}
\tikzset{vertical custom shading/.code={%
 \pgfmathsetmacro\tikz@vcs@middle{#1}
 \pgfmathsetmacro\tikz@vcs@bottom{\tikz@vcs@middle/2}
 \pgfmathsetmacro\tikz@vcs@top{(100-\tikz@vcs@middle)/2+\tikz@vcs@middle}
\pgfdeclareverticalshading[tikz@axis@top,tikz@axis@middle,tikz@axis@bottom]{newaxis}{100bp}{%
  color(0bp)=(tikz@axis@bottom);
  color(\tikz@vcs@bottom bp)=(tikz@axis@bottom);
  color(\tikz@vcs@middle bp)=(tikz@axis@middle);
  color(\tikz@vcs@top bp)=(tikz@axis@top);
  color(100bp)=(tikz@axis@top)}
  \pgfkeysalso{/tikz/shading=newaxis}
  }
}

% commandes pour la premier page 
\newcommand*{\titleCouverture}[1]{\setkeys[PRE]{fam}{#1}

	\begingroup
	\BgThispage
	% Remplacer le logo par rien s'il n'est pas spécifié
	\ifthenelse{\equal{\cmdPRE@fam@logo}{7}}{}{\includegraphics[height=2.5cm,width=20cm,keepaspectratio]{\cmdPRE@fam@logo}\\}
		{\cmdPRE@fam@studies}
	\begin{center}
	\vspace*{\baselineskip}
	{\bfseries \cmdPRE@fam@course}\\[\baselineskip]
	\rule{\textwidth}{1.6pt}\vspace*{-\baselineskip}\vspace*{2pt}
	\rule{\textwidth}{0.4pt}\\[\baselineskip]
	{\Huge \cmdPRE@fam@title}\\[\baselineskip]
	\rule{\textwidth}{0.4pt}\vspace*{-\baselineskip}\vspace{3.2pt}
	\rule{\textwidth}{1.6pt}\\[\baselineskip]
        {\Large \cmdPRE@fam@memoiretitle}\\[\baselineskip]
        {\large \cmdPRE@fam@specialite}\\[\baselineskip]
        %{\Large MÉMOIRE de FIN D’ÉTUDES de MASTER en INFORMATIQUE}\\[\baselineskip]  % Ajoutez 
	\ifthenelse{\equal{\cmdPRE@fam@manyAuthor}{8}}{
	{\Large \cmdPRE@fam@author}\\[\baselineskip]
	{\bfseries \cmdPRE@fam@teacher}}{
	    \begin{minipage}{0.4\textwidth}
	        \begin{flushleft} \large
	        \cmdPRE@fam@author
	        \end{flushleft}
	    \end{minipage}
	    \begin{minipage}{0.4\textwidth}
	        \begin{flushright} \large
	        \cmdPRE@fam@teacher
	        \end{flushright}
	    \end{minipage}
	}
	\vfill
	{\cmdPRE@fam@date}
	\end{center}
	\vspace*{3\baselineskip}
	\pagenumbering{gobble}
	\newpage
	\tableofcontents
    \newpage
    \pagenumbering{arabic}
\endgroup}

% 
\renewcommand{\DOCH}{%
    \settowidth{\py}{\CNoV\thechapter}%
    \addtolength{\py}{-10pt}%     % 
%                                  
    \fboxsep=0pt%
    \colorbox[gray]{.85}{\rule{0pt}{40pt}\parbox[b]{\textwidth}{\hfill}}%
\kern-\myhi
    \kern-\py\raise20pt%
    \hbox{\color[gray]{.5}\CNoV\thechapter}%
%\k
\\%
  }

\makeatother

% environement box 
\newenvironment{frshaded*}{%
	\def\FrameCommand{\fboxrule=\FrameRule\fboxsep=\FrameSep \fcolorbox{framecolor}{shadecolor}}%
	\MakeFramed {\advance\hsize-\width \FrameRestore}}%
{\endMakeFramed}

%Commande pour mettre le logo de ifi Background sur la pge de couverture
\newcommand{\sceau}[1]{\backgroundsetup{
scale=1,
color=black,
opacity=0.05,
angle=10,
position={12cm,-22cm},
contents={%
  \includegraphics[height=20cm,width=20cm,keepaspectratio]{#1}
  }%
}}




% begin: 
\DeclareFieldFormat{labelalphawidth}{\mkbibbrackets{#1}}
 
\defbibenvironment{bibliography}
   {\list
      {\printtext[labelalphawidth]{%
         \printfield{prefixnumber}%
     \printfield{labelalpha}%
         \printfield{extraalpha}}}
      {\setlength{\labelwidth}{\labelalphawidth}%
       \setlength{\leftmargin}{\labelwidth}%
       \setlength{\labelsep}{\biblabelsep}%
       \addtolength{\leftmargin}{\labelsep}%
       \setlength{\itemsep}{\bibitemsep}%
       \setlength{\parsep}{\bibparsep}}%
       \renewcommand*{\makelabel}[1]{##1\hss}}
   {\endlist}
   {\item}
% end: 