(TeX-add-style-hook
 "report4"
 (lambda ()
   (setq TeX-command-extra-options
         "-shell-escape")
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "a4paper")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("babel" "english") ("inputenc" "utf8x") ("geometry" "a4paper" "top=3cm" "bottom=2cm" "left=3cm" "right=3cm" "marginparwidth=1.75cm") ("todonotes" "colorinlistoftodos") ("hyperref" "colorlinks=true" "allcolors=blue")))
   (add-to-list 'LaTeX-verbatim-environments-local "lstlisting")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "babel"
    "inputenc"
    "float"
    "url"
    "helvet"
    "geometry"
    "amsmath"
    "graphicx"
    "todonotes"
    "hyperref"
    "listings")
   (LaTeX-add-labels
    "fig:augmented_command_loss")
   (LaTeX-add-bibitems
    "imitation"))
 :latex)

