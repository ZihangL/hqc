all: note.pdf

note.pdf: note.tex refs.bib
	pdflatex note.tex
	bibtex note.aux
	pdflatex note.tex
	pdflatex note.tex

.PHONY: clean
clean:
	rm -f *.aux *.log *.out *.bbl *.blg
