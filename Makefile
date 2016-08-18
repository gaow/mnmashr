.PHONY: clean install docs

install:
	#@R CMD check ./ --no-manual -o $(shell mktemp -d tmp.XXXX)
	@R CMD build ./ --no-manual
	@((R CMD INSTALL m2ashr_*.tar.gz -l $(shell echo "cat(.libPaths()[1])" | R --slave) && rm -rf tmp.* m2ashr_*.tar.gz) || ($(ECHO) "Installation failed"))

docs:
	@R --vanilla --silent -e 'roxygen2::roxygenise()'

clean:
	rm -f src/m2ash.o src/m2ashr.so src/symbols.rds m2ash_*.tar.gz *.log
