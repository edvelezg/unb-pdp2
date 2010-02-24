. "$TM_SUPPORT_PATH/lib/webpreview.sh"
html_header "Running make"

make ${TM_MAKE_FLAGS} 2>&1|"${TM_RUBY:-ruby}" -rtm_parser -eTextMate.parse_errors

echo "Done."
html_footer