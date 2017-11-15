#! /bin/bash

gs -dDetectDuplicateImages=true -dBATCH -dNOPAUSE -sDEVICE=pdfwrite  -dPDFSETTINGS=/ebook -dCompatibilityLevel=1.4 -sOutputFile=tmp.pdf $1
pdftk tmp.pdf cat output $1
rm tmp.pdf
