/**
 * @param mimeType For SGF file, use "application/x-go-sgf"
 */
const downloadString = (content: string, filename: string, mimeType = 'text/plain'): void => {
    // A WebView ignores <a download> when it points at a blob, so in the app the file goes
    // through the bridge instead, and the user picks where to write it.
    if (window.Native) {
        window.Native.save(filename, mimeType, content);
        return;
    }

    const a = document.createElement('a');
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    a.setAttribute('href', url);
    a.setAttribute('download', filename);
    a.click();
};

export {
    downloadString,
};
