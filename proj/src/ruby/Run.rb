# PROJECT=$(ruby -- "${TM_BUNDLE_SUPPORT}/bin/find_xcode_project.rb")
# if [[ -f "${PROJECT}/project.pbxproj" ]]; then
#    "${TM_BUNDLE_SUPPORT}/bin/run_xcode_target.rb" -project_dir="$PROJECT"
# else
#    echo "Didn't find an Xcode project file."
#    echo "You may want to set TM_XCODE_PROJECT."
# fi

. "$TM_SUPPORT_PATH/lib/webpreview.sh"
html_header "Running make"

make ${TM_MAKE_FLAGS} 2>&1|"${TM_RUBY:-ruby}" -rtm_parser -eTextMate.parse_errors

./PrefixSum 2>&1|"${TM_RUBY:-ruby}" -rtm_parser -eTextMate.parse_errors

echo "Done."
html_footer
