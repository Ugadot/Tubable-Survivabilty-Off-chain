% \ProvidesClass{cup-ino}[2021/03/10 Cambridge University Press, International Organization Journal submission template v1.4]

\newif\if@suppmat\@suppmattrue
\DeclareOption{nosupp}{\@suppmatfalse}
\ProcessOptions\relax
\ExecuteOptions{}
\LoadClass[twoside,10pt]{article}
\RequirePackage{graphicx}
\RequirePackage{etoolbox}
\RequirePackage{regexpatch}
\RequirePackage{ifxetex,ifluatex}

\newif\ifxetexorluatex
\ifxetex
  \xetexorluatextrue
\else
  \ifluatex
    \xetexorluatextrue
  \else
    \xetexorluatexfalse
  \fi
\fi

\RequirePackage{microtype}

\ifxetexorluatex
  \RequirePackage[libertine]{newtxmath}
  \RequirePackage[no-math]{fontspec}
  \setmainfont{TeX Gyre Termes}
\else
  \RequirePackage[utf8]{inputenc}
  \RequirePackage[T1]{fontenc}
  \RequirePackage{newtxtext,newtxmath}
\fi

\RequirePackage{geometry}
\RequirePackage{fancyhdr}
\RequirePackage{lastpage}
\RequirePackage[medium,small,raggedright,noindentafter]{titlesec}
\RequirePackage[runin]{abstract}
\RequirePackage{booktabs}
\RequirePackage{caption,subcaption}
\RequirePackage{flafter}
\RequirePackage[norule,multiple]{footmisc}
\RequirePackage{enumitem}
\RequirePackage[figuresright]{rotating}
\RequirePackage{lineno}

\geometry{paperwidth=6in,paperheight=9in,
  top=1in,bottom=0.75in,inner=0.75in,outer=0.6in,
  headsep=0.3in,footskip=0.3in,footnotesep=0.4in,
  reversemarginpar,marginparwidth=0.575in,marginparsep=0.5em}
  
\newcommand{\runningtitle}[1]{\def\@runningtitle{#1}}
% \def\@journalname{International Organization}

\renewcommand{\@maketitle}{%
  \begingroup\raggedright%
  \rule{\textwidth}{1pt}\par
  {\LARGE\bfseries\@title\par}
  \medskip
  {\Large\@author\par\vspace*{1em}}
  \endgroup
}

\renewcommand{\maketitle}{\vspace*{\dimexpr-\headsep-\baselineskip}\@maketitle\thispagestyle{firstpage}}

\titlespacing{name=\section}{0pt}{2\baselineskip}{\baselineskip}
\titlespacing{name=\section,numberless}{0pt}{2\baselineskip}{0.5em}
  
\fancyhf{}
\renewcommand{\headrulewidth}{1pt}
\fancyhead[LO]{\textbf{\thepage}\quad\@runningtitle}
\fancyhead[LE]{\textbf{\thepage}\quad\@runningtitle}
\pagestyle{fancy}
\fancypagestyle{firstpage}{%
  \fancyhf{}
  \renewcommand{\headrule}{}
}

\setlength{\abstitleskip}{-\parindent}
\abslabeldelim{\quad}
\setlength{\absleftindent}{1em}
\setlength{\absrightindent}{1em}
\preto{\abstract}{%
  {\noindent\rule{\textwidth}{1pt}}\vspace*{1em}%
}
\appto{\endabstract}{%
  \vspace*{1em}%
  {\noindent\raisebox{1em}{\rule{\textwidth}{1pt}}}\vspace*{\baselineskip}
}

% \preto{\enddocument}{%
%   {\bigskip\noindent\abstracttextfont Date received: MMMM DD, YYYY; Date accepted: MMMM DD, YYYY. \par}\marginpar{\scriptsize\raggedleft Dummy dates; please ignore.}%
% }

\setcounter{secnumdepth}{0}
\setlist{nosep,leftmargin=*,align=left,widest=00}
\newenvironment{authorbio}{\setlength{\parindent}{\z@}}{}

\captionsetup*{labelfont={bf,footnotesize},textfont={it},labelsep={period},justification=centering,singlelinecheck=false}
\captionsetup*[figure]{name=FIGURE}
\captionsetup*[table]{name=TABLE,position=top,justification=raggedright}
\newcommand{\floatnotes}[2][Notes: ]{%
  \par\medskip%
  \begingroup%
  \begin{minipage}{0.9\linewidth}
  \raggedright\footnotesize%
  \emph{#1}#2\par
  \end{minipage}\endgroup%
}

\RequirePackage[backend=biber,authordate,autocite=footnote,dashed=false,language=british]{biblatex-chicago}
\ExecuteBibliographyOptions[jurisdiction,legal,legislation]{skipbib=false}
\DeclareFieldFormat[article]{title}{#1\isdot}
\DeclareFieldFormat[incollection]{title}{#1\isdot}
\DeclareFieldFormat[inbook]{title}{#1\isdot}
\DeclareFieldFormat[report]{title}{#1\isdot}
\DeclareFieldFormat[inproceedings]{title}{#1\isdot}
\DeclareFieldFormat[inproceedings]{booktitle}{#1\isdot}
\DeclareFieldFormat[thesis]{title}{#1\isdot}
\DeclareFieldFormat[misc]{title}{#1\isdot}
\DeclareFieldFormat{url}{Available at <\url{#1}>}
\DefineBibliographyStrings{english}{%
  in = {\ifentrytype{inproceedings}{Paper presented at the}{In}},
}
\AtEveryBibitem{%
  \ifentrytype{book}{\clearlist{location}}{}
}
\urlstyle{same}

\xpatchbibmacro{bib+doi+url}
  {\printurldate}{}{}{}
\xapptobibmacro{bib+doi+url}{%
  \newunit\newblock%
  \ifboolexpr{%
    togl {cms@url}%
    and
    not test {\iffieldundef{urlyear}}%
    }%
  {\printurldate}% Date fix
  {}%
}{}{}
%% Remove space after colon of numbers, before pages
\xpatchbibmacro*{journal+issue+year+pages}
  {\setunit{\addcolon\addspace}}
  {\setunit{\addcolon}}
  {}{}

% v1.3: Oxford commas
\AtBeginDocument{\renewcommand{\finalandcomma}{\addcomma}}

\setlength{\bibhang}{1em}
\renewcommand{\bibfont}{\small}
\setlength{\bibitemsep}{\z@}

\newcommand{\insert@supplmat}{%
\if@suppmat
  \section{Supplementary Material}
  Simple python implementation for the algorithm:\\  \url{https://github.com/Ugadot/Tubable-Survivabilty-Off-chain}
  
\fi\@suppmatfalse
}
\xpretocmd{\printbibliography}{\insert@supplmat}{}{}
\AtBeginEnvironment{thebibliography}{\insert@supplmat}

\appto{\appendix}{%
  \insert@supplmat
  \setcounter{table}{0}
  \setcounter{figure}{0}
  \setcounter{equation}{0}
  \renewcommand{\thefigure}{A\arabic{figure}}
  \renewcommand{\thetable}{A\arabic{table}}
  \renewcommand{\theequation}{A\arabic{equation}}
  \section*{Appendix}
}

%% Tables use smaller font sizes
\DeclareRobustCommand{\patchINOtable}[1]{%
  \BeforeBeginEnvironment{#1}{\footnotesize}
}
%% Heavy table rules
\renewcommand{\belowrulesep}{1em}
\renewcommand{\aboverulesep}{1em}
\setlength{\heavyrulewidth}{2pt}
\AtBeginDocument{%
  \patchINOtable{tabular}
  \patchINOtable{tabular*}
  \patchINOtable{sidewaystable}
  \patchINOtable{sidewaystable*}
  \@ifpackageloaded{tabularx}{\patchINOtable{tabularx}}{}
  \@ifpackageloaded{supertabular}{\patchINOtable{supertabular}}{}
  \@ifpackageloaded{longtable}{\patchINOtable{longtable}\setlength{\LTleft}{\z@}}{}
%   \linenumbers
  \ifdefvoid{\@runningtitle}{\edef\@runningtitle{\@title}}{}
}

\AtEndPreamble{
  \RequirePackage[hyperfootnotes=false]{hyperref}
}