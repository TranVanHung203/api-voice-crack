Set shell = CreateObject("WScript.Shell")
Set filesystem = CreateObject("Scripting.FileSystemObject")

shell.CurrentDirectory = filesystem.GetParentFolderName(WScript.ScriptFullName)
shell.Run "cmd /k py -3.11 -m uvicorn server:app --host 0.0.0.0 --port 8000", 1, False
