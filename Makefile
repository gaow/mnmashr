.PHONY: clean install docs

install:
	@R --vanilla --silent -e 'Rcpp::compileAttributes()'
	#@R CMD check ./ --no-manual -o $(shell mktemp -d tmp.XXXX)
	@R CMD build ./ --no-manual
	@((R CMD INSTALL mnmashr_*.tar.gz -l $(shell echo "cat(.libPaths()[1])" | R --slave) && rm -rf tmp.* mnmashr_*.tar.gz) || ($(ECHO) "Installation failed"))

docs:
	@R --vanilla --silent -e 'roxygen2::roxygenise()'

clean:
	rm -f src/mnmash.o src/mnmashr.so src/symbols.rds mnmashr_*.tar.gz *.log
