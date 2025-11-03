; Set the hotkey to Ctrl + J
^j::
    ; Set the path to the folder containing your script
    working_dir := "C:\\Users\\henry\\OneDrive\\Documents\\My Code\\Second Brain"
    python_script := "SecondBrainFrontend.py" ; Can be just the filename now

    ; Run the script, setting the working directory first
    Run, pythonw.exe "%python_script%", %working_dir%

    return