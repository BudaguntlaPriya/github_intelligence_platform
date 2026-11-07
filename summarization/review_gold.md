# Manual Priority Gold Set Audit (All 130 Issues)

This document contains the complete, audited draft priority labels for all **130 issues** in the gold evaluation set (60 random + 30 severity-enriched + 40 metadata-sampled: 20 comments >= 8, 10 label bug/blocker, 10 created >= 2021).

**Sorted by Confidence: LOW-confidence (borderline / debatable cases) are placed FIRST for your careful review.**

> [!IMPORTANT]
> **Strict Definitions Applied**:
> - **Critical**: Breaks installs/startup for many users, exploitable vulnerability in Flask core itself, or silent data loss. (None found in this 130-issue sample; see notes below).
> - **High**: Core functionality broken, severe regressions affecting many users without workaround, or unhandled crashes in core pipelines.
> - **Medium**: Non-critical bugs, edge cases, minor regressions with workarounds, or deprecations.
> - **Low**: Docs, tutorials, typos, questions, feature requests, unsupported setups, external extensions.

## Summary Distribution of Audited Draft Labels
- **Total Issues**: 130
- **Priority Breakdown**: Low: 101 (77.7%) | Medium: 24 (18.5%) | High: 5 (3.8%) | Critical: 0 (0.0%)
- **Confidence Breakdown**: Low (Borderline): 19 | Medium: 13 | High (Clear): 98

## All 130 Issues (Sorted LOW-Confidence First)

| # | Issue | Sample | Draft Priority | Confidence | One-Sentence Rationale (Full Text & Comments) |
|---|---|---|:---:|:---:|---|
| 1 | **#248**: Review JSON Security | `severity_enriched` | **Low** | `Low` | Legacy 2011 documentation review regarding ECMAScript 5 top-level array literals and historical JSON hijacking in outdated browsers. |
| 2 | **#367**: support for HTTP/1.1 Transfer-Encoding: chunked | `random` | **High** | `Low` | Protocol non-compliance defect where incoming HTTP/1.1 chunked transfer encoded requests fail during JSON request parsing. |
| 3 | **#651**: Cannot start Flask application in a thread other than main | `random` | **Medium** | `Low` | Unhandled crash when running the development server in a non-main worker thread because Python signal handlers are restricted to the main thread. |
| 4 | **#953**: flaskr Stored XSS [FIX] | `random` | **Medium** | `Low` | Stored cross-site scripting vulnerability in the tutorial example blog application (examples/flaskr), not in Flask core framework itself. |
| 5 | **#1177**: Should jsonify() throw an exception when provided a list? | `severity_enriched` | **Low** | `Low` | Discrepancy between documentation advising against top-level JSON arrays and jsonify implementation permitting them in Python dictionaries. |
| 6 | **#1437**: Flask won't start in the Python 3.4 debugger | `random` | **Medium** | `Low` | Compatibility failure preventing Flask applications from launching under the standard Python 3.4 interactive debugger (pdb3). |
| 7 | **#2188**: Input stream not automatically drained at teardown. | `random` | **High** | `Low` | Unconsumed request body data remains in the socket buffer under HTTP keep-alive, corrupting subsequent requests with 400 Bad Request. |
| 8 | **#2927**: '301' is not a recognized HTTP error code | `random` | **Medium** | `Low` | Migration issue across major upgrade where registering an error handler on an HTTP redirect status code (301) fails because it is not an exception. |
| 9 | **#3223**: AttributeError: 'NoneType' object has no attribute 'append' in Flask helpers.py | `random` | **Medium** | `Low` | Intermittent AttributeError on logout during message flashing that could not be reproduced by maintainers and likely stems from custom session backends. |
| 10 | **#3291**: Custom Response force_type is now never called | `random` | **Low** | `Low` | User complained custom response force_type was not invoked, but maintainers clarified force_type is strictly meant for WSGI callables, not data models. |
| 11 | **#3761**: A Flask app holds a reference to itself | `random` | **Low** | `Low` | Circular memory reference between Flask app and its static route that is safely handled and reclaimed by Python's cyclic garbage collector. |
| 12 | **#3874**: flask doesn't return http error 413 if file too large | `severity_enriched` | **Low** | `Low` | Development server connection reset instead of HTTP 413 on oversized file uploads, which maintainers documented as normal socket behavior for the dev server. |
| 13 | **#4023**: Flask no longer provides `flask._compat` module | `created_2021_or_later` | **Medium** | `Low` | Removal of internal flask._compat in Flask 2.0 broke unmaintained third-party ecosystem tools like flask-script. |
| 14 | **#4494**: Flask failing to startup due to Jinja2 breaking change | `comments_ge_8` | **Medium** | `Low` | Jinja2 3.1.0 dropped an import that causes startup crashes for users running legacy unpinned Flask 1.1.x. |
| 15 | **#4725**: Flask test does not work after 2.2.0 update (with sqlalchemy dependency) | `random` | **Low** | `Low` | Test runner failure reported after 2.2.0 update that maintainers could not reproduce and traced to third-party SQLAlchemy configuration. |
| 16 | **#4739**: Tests failing on latest Flask version 2.2.1 | `random` | **Medium** | `Low` | Test runner failures on tox and pytest doctest following release, which was closed as an upstream regression in Werkzeug #2485. |
| 17 | **#5256**: AsyncToSync error with Flask Async, Gunicorn and Gevent | `random` | **Low** | `Low` | Race condition under heavy concurrency with async teardown, Gevent, and Gunicorn that maintainers confirmed originates in Gevent and asgiref rather than Flask. |
| 18 | **#5513**: Handler `before_request` doesn't work as expected. | `random` | **Low** | `Low` | User attempt to implement custom CORS pre-flight handling via before_request failed because OPTIONS requests require an immediate returned response. |
| 19 | **#5870**: Teardown handler chain exception handling | `severity_enriched` | **Medium** | `Low` | Architectural limitation where an unhandled exception in one teardown callback halts execution of subsequent teardown callbacks, risking resource leaks. |
| 20 | **#824**: url_for(*args, _external=True) doesn't add the port to the full url | `comments_ge_8` | **Medium** | `Medium` | url_for with _external=True omits the non-standard port number when SERVER_NAME is not explicitly configured. |
| 21 | **#968**: GET params discarded by flask test client, when using full url | `random` | **Medium** | `Medium` | Test client URL parser bug silently discarding query string parameters when passed an absolute URL rather than a relative path. |
| 22 | **#974**: jsonify() ignores the sort order of an OrderedDict | `comments_ge_8` | **Medium** | `Medium` | JSON serializer sorts keys alphabetically by default, overriding the user's explicit OrderedDict sequence. |
| 23 | **#1427**: Bug in send_file with temp files | `label_bug_blocker` | **High** | `Medium` | send_file fails under Python 3 when passed a temporary file object because of missing filename/mtime handling. |
| 24 | **#1955**: Flask Test Client fails to get on subdomains with explicit port 80 | `random` | **Medium** | `Medium` | Test client fails to resolve subdomain routes when an explicit default port (:80) is configured in SERVER_NAME. |
| 25 | **#2005**: AttributeError: 'DispatchingApp' object has no attribute 'config' | `label_bug_blocker` | **Medium** | `Medium` | Unhandled AttributeError in tutorial example on master branch when running via flask run. |
| 26 | **#2616**: A setup function was called after the first request was handled | `severity_enriched` | **Low** | `Medium` | Invalid application architecture attempting to register blueprint routes dynamically after the server has already started handling requests. |
| 27 | **#3275**: flask requires `is_package` from import loaders, however this api is optional since PEP 451 | `random` | **Medium** | `Medium` | Compatibility defect where Flask's module loader assumes is_package is always present, violating PEP 451 loader specifications. |
| 28 | **#3330**: test_client.set_cookie does not seem to work | `severity_enriched` | **Medium** | `Medium` | User confusion between session cookies and direct test_client cookie handling that revealed confusing cookie management semantics in testing. |
| 29 | **#3469**: Cannot set SameSite=None on session cookie | `random` | **Medium** | `Medium` | Inability to set SameSite=None on session cookies due to cookie formatting in Werkzeug, closed as duplicate of Werkzeug #1549. |
| 30 | **#4025**: Flask 1.1.2 accidentally ships with `itsdangerous` v2, hence ignores `simplejson` | `created_2021_or_later` | **Medium** | `Medium` | Transitive dependency conflict where Flask 1.1.2 unpinned itsdangerous v2, unexpectedly breaking simplejson serialization. |
| 31 | **#4043**: Older versions (1.x.x) of flask pin some dependencies in a way that could cause issues | `severity_enriched` | **Low** | `Medium` | Dependency version pinning discussion regarding semantic version ranges on the legacy end-of-life 1.x branch. |
| 32 | **#5337**: Incompatible return value type: got "werkzeug.wrappers.response.Response", expected "flask.wrappers.Response" | `severity_enriched` | **Medium** | `Medium` | Static typing regression in Flask 3.0 where response wrapper return type annotations diverged from Werkzeug response objects. |
| 33 | **#100**: Inconsistency in docs/testing.rst | `severity_enriched` | **Low** | `High` | Documentation inconsistency in test fixture variable naming in the testing guide. |
| 34 | **#143**: Pointless example for Signal.connected_to | `random` | **Low** | `High` | Documentation improvement pointing out an ineffective code snippet in the signals tutorial. |
| 35 | **#177**: Tutorial should mention deployment with mod_wsgi or something else | `random` | **Low** | `High` | Documentation request to expand the tutorial with deployment guidance for mod_wsgi. |
| 36 | **#216**: flask has no programmatically-accessible version number | `random` | **Low** | `High` | Feature request proposing programmatic version attributes (__version__ or version_info). |
| 37 | **#241**: Tutorial Step 4 : teardown code sample is not well formated | `severity_enriched` | **Low** | `High` | Documentation formatting defect where code indentation was broken on the database tutorial page. |
| 38 | **#323**: SERVER_NAME set to 127.0.0.1:5000 in dev breaks flash method | `comments_ge_8` | **Low** | `High` | Cookie domain issue on legacy WebKit browsers when specifying IP and port in SERVER_NAME. |
| 39 | **#378**: jsonify should set the charset | `comments_ge_8` | **Low** | `High` | Proposal to explicitly set charset=utf-8 on application/json responses for Internet Explorer 9 compliance. |
| 40 | **#381**: Consider debug=True in test client. | `random` | **Low** | `High` | Ergonomic feature proposal to allow setting debug=True directly on the test client for view function assertions. |
| 41 | **#399**: Collab me! | `random` | **Low** | `High` | Non-technical casual submission asking for repository collaborator permissions with emoticons. |
| 42 | **#481**: Add a FORCE_URL_SCHEME option | `severity_enriched` | **Low** | `High` | Enhancement proposal to provide a global configuration option forcing the HTTPS URL scheme. |
| 43 | **#587**: Python 3 port is available | `comments_ge_8` | **Low** | `High` | Community status announcement and patch submission concerning Python 3 porting progress. |
| 44 | **#593**: Nestable blueprints | `comments_ge_8` | **Low** | `High` | Feature request proposing support for nested blueprints with parent-child registration (later added in 2.0). |
| 45 | **#631**: Add config parser to blueprints | `random` | **Low** | `High` | Feature request proposing blueprint-scoped configuration parsing merged into the application on registration. |
| 46 | **#639**: Get HTTP_REFERER in flask | `random` | **Low** | `High` | General user inquiry asking how to retrieve the HTTP_REFERER header from incoming requests. |
| 47 | **#686**: Raise 400 JSONBadRequest instead of KeyError when accessing JSON request data | `random` | **Low** | `High` | API design enhancement proposing a custom JSONBadRequest exception rather than KeyError for missing JSON payload keys. |
| 48 | **#757**: Is there best solution for http client connection pool? | `random` | **Low** | `High` | General architecture question asking for recommendations on outbound HTTP client connection pooling. |
| 49 | **#760**: blueprint.route methods argument typo throws unclear exception | `comments_ge_8` | **Low** | `High` | Exception message enhancement to provide clearer error output when a typo is made in the methods parameter. |
| 50 | **#857**: Exception when trying to run Flask app via CGIHandler | `comments_ge_8` | **Low** | `High` | User configuration issue attempting to run Flask applications under Apache 2 CGIHandler. |
| 51 | **#903**: How to generate a SecureCookie in latest version? | `severity_enriched` | **Low** | `High` | Migration inquiry regarding API movement of SecureCookie across Werkzeug releases. |
| 52 | **#925**: Message Flashing's implementation is broken and should be removed anyway | `comments_ge_8` | **Low** | `High` | Architecture discussion and debate concerning message flashing behavior across browser tabs. |
| 53 | **#939**: Default HTML errors don't tell their encoding and aren't HTML5 | `random` | **Low** | `High` | Standards compliance enhancement requesting HTML5 doctype and explicit charset encoding on default error pages. |
| 54 | **#941**: Registering a handler for HTTPException has no effect | `label_bug_blocker` | **High** | `High` | Registering a custom error handler for HTTPException has no effect because Flask internally overrides standard HTTP error dispatching. |
| 55 | **#976**: "import os" statement missing from tutorial | `random` | **Low** | `High` | Minor documentation fix adding a missing import os statement to the tutorial setup code snippet. |
| 56 | **#1011**: Python 3.4 changed `pkgutil.get_loader` behavior | `label_bug_blocker` | **High** | `High` | Causes application startup crash during Flask(__name__) on Python 3.4 because pkgutil.get_loader behavior changed in Python 3.4. |
| 57 | **#1073**: Python Flask Gevent stack - Simple “Hello World” app shows as inefficient when benchmarked | `comments_ge_8` | **Low** | `High` | General performance benchmark inquiry comparing Flask and Gevent against raw WSGI servers. |
| 58 | **#1188**: [Website] Documentation PDF and zipped HTML download links return 404 | `random` | **Low** | `High` | Dead external hyperlinks to legacy PDF and zipped HTML documentation bundles on the project website. |
| 59 | **#1272**: Time for a new release ? | `random` | **Low** | `High` | Community inquiry asking maintainers about release planning and timeline for the next version tag. |
| 60 | **#1470**: route function quit unexpectedly | `random` | **Low** | `High` | Segmentation fault or process crash originating inside a compiled third-party C++ extension wrapper. |
| 61 | **#1471**: Adding after_server_run | `random` | **Low** | `High` | Feature request proposing a new post-startup lifecycle callback hook (after_server_run). |
| 62 | **#1497**: Feedback Links - 404 Error | `random` | **Low** | `High` | Dead external hyperlinks to a deprecated community feedback portal. |
| 63 | **#1506**: flash(...) won't work! | `random` | **Low** | `High` | Standard user configuration mistake calling flash() without defining the required secret_key on the application. |
| 64 | **#1541**: Returned 200 for illegal request url | `comments_ge_8` | **Low** | `High` | User sending malformed HTTP request line (foo://...) via netcat directly to the development server. |
| 65 | **#1545**: Request.form data dict... Could we modify the structure? | `random` | **Low** | `High` | Feature proposal to automatically parse bracketed form field keys (item[1][name]) into nested dictionaries. |
| 66 | **#1548**: Support zero-ceremony sub-app mounting (including Blueprints) | `comments_ge_8` | **Low** | `High` | API design proposal to support zero-ceremony sub-application mounting alongside blueprints. |
| 67 | **#1578**: Update docs on the website (at least the python 3 ones) | `comments_ge_8` | **Low** | `High` | Documentation request to update outdated Python 3 guidance on the project website. |
| 68 | **#1656**: Injecting Flask vulnerability issue | `severity_enriched` | **Low** | `High` | Documentation inquiry regarding an external blog post discussing server-side template injection in application code. |
| 69 | **#1787**: database: persistent connetion | `random` | **Low** | `High` | General support inquiry asking where to store persistent database connections across application lifecycles. |
| 70 | **#1877**: jsonify no longer causes Content-Length to be set on returned Response object | `label_bug_blocker` | **Medium** | `High` | Regression where adding a newline in jsonify caused the Content-Length response header to no longer be set. |
| 71 | **#1893**: Travis 3.3-devel job fails with error about urllib2 in Werkzeug | `label_bug_blocker` | **Low** | `High` | Continuous integration build failure on Python 3.3-dev due to urllib2 import changes in Werkzeug. |
| 72 | **#1916**: Document option to use simplejson | `comments_ge_8` | **Low** | `High` | Documentation request to document the optional simplejson dependency and configuration flags. |
| 73 | **#1919**: Remove easy_install from installation docs | `random` | **Low** | `High` | Documentation cleanup removing references to the deprecated easy_install packaging utility. |
| 74 | **#1931**: dialogue box? | `random` | **Low** | `High` | Unactionable issue submission with an empty description and vague single-word title. |
| 75 | **#1933**: JSON Parsing on empty request content with RequestParser | `random` | **Low** | `High` | Third-party library integration report referencing an external flask-restful issue tracker without a Flask core defect. |
| 76 | **#2027**: Update app factory docs for Flask's CLI | `random` | **Low** | `High` | Documentation enhancement request to update the CLI documentation to reflect application factory patterns. |
| 77 | **#2030**: Support same-site cookies | `random` | **Low** | `High` | Forward-looking security enhancement proposing support for SameSite cookie attributes based on a new RFC. |
| 78 | **#2058**: The file/path provided (flaskr) does not appear to exist. | `comments_ge_8` | **Low** | `High` | User path mistake encountering CLI error when attempting to run the tutorial application. |
| 79 | **#2421**: Tutorial: step 3 (Installing as package) needs rationale | `comments_ge_8` | **Low** | `High` | Documentation improvement requesting rationale for packaging recommendations in tutorial step 3. |
| 80 | **#2456**: The request method to another website in views function didn't run correctly | `random` | **Low** | `High` | User application implementation mistake using the external requests library inside a route function, not a Flask defect. |
| 81 | **#2479**: Flask crashes on startup | `severity_enriched` | **Low** | `High` | Single-user Anaconda Windows execution environment error attempting to run a binary launcher as a Python text script. |
| 82 | **#2514**: Unable to generate the PDF version of the documentation | `severity_enriched` | **Low** | `High` | Build failure in the Sphinx LaTeX documentation toolchain when compiling PDF documentation. |
| 83 | **#2549**: "Request.is_xhr deprecated" warning during jsonify() | `label_bug_blocker` | **Medium** | `High` | Flask's internal jsonify() implementation accesses deprecated Request.is_xhr, triggering deprecation warnings on every JSON response. |
| 84 | **#2655**: CSRF session Token is missing(400 Bad Request) | `random` | **Low** | `High` | Invalid issue submission because CSRF protection is provided by third-party extensions (Flask-WTF), not Flask core. |
| 85 | **#2715**: E-commerce | `random` | **Low** | `High` | General ecosystem discussion proposing that the Flask community port the Django Oscar e-commerce framework. |
| 86 | **#2743**: Documentation search broken | `label_bug_blocker` | **Low** | `High` | Sphinx documentation search hyperlink defect on the project website generating broken 404 links. |
| 87 | **#2838**: Logging example fails outside request context | `comments_ge_8` | **Low** | `High` | Documentation example in the logging guide accessed request.url without checking has_request_context(). |
| 88 | **#2882**: RuntimeError: can't start new thread, Exception happened during processing of request from | `severity_enriched` | **Low** | `High` | Operating system resource exhaustion error attempting to spawn worker threads on an overloaded machine. |
| 89 | **#2911**: Flaskr tutorial: auth.py raise a type error: tuple indices must be integers not str | `random` | **Low** | `High` | Tutorial user implementation mistake omitting the sqlite3.Row row factory configuration step. |
| 90 | **#3055**: How to connect SQL sever/MSSQL database with flask are there any libraries available | `severity_enriched` | **Low** | `High` | General support question asking for ecosystem database driver recommendations for Microsoft SQL Server. |
| 91 | **#3131**: can't run app with latest Werkzeug | `random` | **Low** | `High` | User misunderstanding of CLI execution commands, resolved when running the development server via flask run. |
| 92 | **#3143**: I am not an expert, but I want to solve this error: AssertionError: View function mapping is overwriting an existing endpoint function: register | `severity_enriched` | **Low** | `High` | User routing error caused by defining duplicate view function names on the same application. |
| 93 | **#3168**: Improve warning about dev server use in production | `comments_ge_8` | **Low** | `High` | Clarity improvement updating the development server startup warning message for beginners (good first issue). |
| 94 | **#3218**: Use @property as a decorator | `random` | **Low** | `High` | Internal code style refactoring proposal to use decorator syntax for class properties. |
| 95 | **#3252**: Should _get_exc_class_and_code method have param entry for the docstring? | `random` | **Low** | `High` | Minor documentation inquiry asking whether to add parameter docstrings to an internal helper method. |
| 96 | **#3318**: Provide secure default hash function for SecureCookieSessionInterface digest_method | `severity_enriched` | **Low** | `High` | Cryptographic proposal to modernize default session cookie HMAC hashing from SHA-1 to SHA-256. |
| 97 | **#3320**: Version tag mismatch in link to example projects from docs | `label_bug_blocker` | **Low** | `High` | Minor documentation link mismatch pointing to a development branch version tag instead of the release tag. |
| 98 | **#3447**: Flask crashes when running as a Windows Service (pywin32) | `random` | **Low** | `High` | Unsupported usage attempting to execute the interactive development server as a headless Windows background service without standard I/O streams. |
| 99 | **#3718**: Testing tutorial doc is not working | `severity_enriched` | **Low** | `High` | Documentation inquiry regarding pytest fixture usage in the testing tutorial. |
| 100 | **#3770**: why flask create a new thread with the same TID that already exists? | `severity_enriched` | **Low** | `High` | User inquiry concerning OS-level thread identifier reuse rather than a defect in Flask. |
| 101 | **#3776**: request.endpoint always None in SessionInterface.open_session | `label_bug_blocker` | **Medium** | `High` | Regression where request.endpoint is always None during SessionInterface.open_session due to routing dispatch order change. |
| 102 | **#3902**: flask.run(post_func)? | `random` | **Low** | `High` | Feature request proposing an automatic browser launch callback on server start, rejected by maintainers. |
| 103 | **#3945**: window server 2019 run sometimes request timeout need press enter | `random` | **Low** | `High` | Windows command prompt QuickEdit mode pausing console execution upon text selection, not a Flask framework bug. |
| 104 | **#3959**: Docs - Testing Flask Applications | `random` | **Low** | `High` | Incomplete and unfilled issue template concerning testing documentation, closed as duplicate of #3545. |
| 105 | **#4054**: Unused import statement in docs /quickstart/#variable-rules code block. | `random` | **Low** | `High` | Documentation cleanup removing an unused import from a quickstart example code block. |
| 106 | **#4099**: Harmless race condition in tutorial | `created_2021_or_later` | **Low** | `High` | Documentation observation noting a harmless race condition in the tutorial registration view. |
| 107 | **#4256**: Flask logger does not work when served with an ASGI server like Uvicorn | `random` | **Low** | `High` | User error overwriting the application instance variable with an external WsgiToAsgi adapter, causing app.logger to become undefined. |
| 108 | **#4288**: 2.0.2: test suite is failing | `severity_enriched` | **Low** | `High` | Packaging build issue caused by invoking test suites outside supported virtual environments and tox harnesses. |
| 109 | **#4291**: MethodView Decorators list execution order | `severity_enriched` | **Low** | `High` | User confusion regarding Python decorator evaluation order rather than a defect in MethodView. |
| 110 | **#4427**: Not sanitising JSON | `severity_enriched` | **Low** | `High` | Invalid report stemming from confusion between Python's internal string repr and actual serialized JSON payloads. |
| 111 | **#4431**: First Tutorial Step Returns 404 | `random` | **Low** | `High` | Tutorial user reproduction mistake missing a directory or factory configuration step. |
| 112 | **#4464**: docs: 'Sending Requests with the Test Client' references non-existent 'query' parameter | `random` | **Low** | `High` | Documentation typo misnaming the test client query_string parameter as query. |
| 113 | **#4599**: Very strange error | `created_2021_or_later` | **Low** | `High` | User connection error attempting to access a plain HTTP development server using an HTTPS URL. |
| 114 | **#4604**: Update of the downloadable documentation of flask2.0 | `severity_enriched` | **Low** | `High` | Feature request asking for pre-built offline downloadable documentation bundles for version 2.0. |
| 115 | **#4650**: There is a typo in the docs relating to javascript | `random` | **Low** | `High` | Minor documentation typo omitting a verb in the JavaScript documentation section. |
| 116 | **#4714**: deprecate `FLASK_ENV` | `severity_enriched` | **Low** | `High` | Architectural proposal to deprecate the redundant FLASK_ENV configuration variable in favor of debug flags. |
| 117 | **#4728**: `Flask test` does not work (tables not created) after 2.2.0 update | `random` | **Low** | `High` | Closed by maintainers as an exact duplicate of #4725 regarding test runner database initialization. |
| 118 | **#4930**: Testing with authenticated and anonymous client at the same time gives mixed results | `created_2021_or_later` | **Low** | `High` | User test fixture configuration problem with flask-login resolved by user after maintainer explanation. |
| 119 | **#4950**: Edit README.rst in examples/tutorial | `random` | **Low** | `High` | Version mismatch between tutorial README on branch main and user's locally installed older Flask release. |
| 120 | **#5021**: Update Flask to work with Werkzeug 2.2.3 due to open CVE | `severity_enriched` | **Low** | `High` | Upstream dependency bump request for a Werkzeug CVE, which was already compatible with Flask core. |
| 121 | **#5031**: URL Hack? | `created_2021_or_later` | **Low** | `High` | User security inquiry about malicious HTTP scans hitting an open, unauthenticated MongoDB instance. |
| 122 | **#5407**: Extend Config type | `random` | **Low** | `High` | Type hinting limitation in Pylance when resolving arbitrary dynamic dictionary keys in app.config. |
| 123 | **#5448**: Use of `hashlib.sha1` causes issue in FIPS build | `severity_enriched` | **Low** | `High` | Maintainer confirmed the traceback is from the external Flask-Session/cachelib extension under FIPS, not Flask core itself. |
| 124 | **#5621**: Rotating secret keys | `created_2021_or_later` | **Low** | `High` | Feature request proposing secret key rotation capabilities (OLD_SECRET_KEYS) for signed cookies. |
| 125 | **#5729**: The `template_filter` decorator doesn't work if you don't pass an argument | `comments_ge_8` | **Medium** | `High` | The @app.template_filter decorator silently fails to register template filters if called without parentheses. |
| 126 | **#5776**: Looser type annotations for send_file() path_or_file argument | `created_2021_or_later` | **Medium** | `High` | Strict type annotations on send_file prevent passing valid IO[bytes] file-like objects supported by Werkzeug. |
| 127 | **#5848**: Add copy-to-clipboard button for code blocks in Flask documentation. | `created_2021_or_later` | **Low** | `High` | Feature request proposing a copy-to-clipboard button for code blocks in Sphinx documentation. |
| 128 | **#5991**: <spam> | `severity_enriched` | **Low** | `High` | Irrelevant commercial promotional spam submission with no technical content. |
| 129 | **#5993**: <spam> | `created_2021_or_later` | **Low** | `High` | Commercial promotional spam submission with removed content. |
| 130 | **#6139**: AI junk | `severity_enriched` | **Low** | `High` | Spurious AI-generated vulnerability report claiming non-existent code execution in configuration loaders. |
