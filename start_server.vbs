Set shell = CreateObject("WScript.Shell")
Set filesystem = CreateObject("Scripting.FileSystemObject")

shell.CurrentDirectory = filesystem.GetParentFolderName(WScript.ScriptFullName)
shell.Run "powershell.exe -NoProfile -NoExit -ExecutionPolicy RemoteSigned -File """ & filesystem.BuildPath(shell.CurrentDirectory, "start_server.ps1") & """", 1, False
