PROJ = main
LC = latexmk
AUXSDIR = .build
OPT = -halt-on-error
ARGS = -pdfxe -outdir=$(AUXSDIR)  -pdfxelatex="xelatex $(OPT)" -use-make -bibtex

all: $(PROJ).pdf

$(PROJ).pdf : $(PROJ).tex 
	$(LC) $(ARGS)
	mv .build/${PROJ}.pdf ./

clean:
	rm -rf $(AUXSDIR)

.PHONY: all clean