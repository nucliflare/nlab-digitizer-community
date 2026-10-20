-- GFM drops custom heading identifiers. Preserve the topic IDs used by the
-- shared offline manual's related links instead of depending on title slugs.
function Header(header)
  if header.identifier ~= "" then
    return {
      pandoc.RawBlock("html", '<a id="' .. header.identifier .. '"></a>'),
      header,
    }
  end
end
