# this command calls xcodebuild giving the name of the project
# directory as the project to build and parses the output for
# file/line information then plays a succes/failure sample
# based on the final outcome
#
# if the build succeeds, it will run the target

# XCODE_PROJECT_FILE=$(ruby -- "${TM_BUNDLE_SUPPORT}/bin/find_xcode_project.rb")
# export XCODE_PROJECT_FILE

# export XCODE_RUN_BUILD=1
# "${TM_BUNDLE_SUPPORT}/bin/run_xcodebuild.sh"

#&& "${TM_BUNDLE_SUPPORT}/bin/run_xcode_target.rb" -project_dir="$XCODE_PROJECT_FILE"

. "$TM_SUPPORT_PATH/lib/webpreview.sh"
html_header "Running make"

make ${TM_MAKE_FLAGS} 2>&1|"${TM_RUBY:-ruby}" -rtm_parser -eTextMate.parse_errors

Debug/PrefixSum 2>&1|"${TM_RUBY:-ruby}" -rtm_parser -eTextMate.parse_errors

echo "Done."
html_footer
