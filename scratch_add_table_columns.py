import re

with open('e:/aitrade/frontend/src/pages/AdminDashboard.tsx', 'r', encoding='utf-8') as f:
    code = f.read()

# Update Table Headers
old_headers = """                      <th className="px-6 py-4 text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">Username</th>
                      <th className="px-6 py-4 text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">Role</th>"""

new_headers = """                      <th className="px-6 py-4 text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">Username</th>
                      <th className="px-6 py-4 text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">Email</th>
                      <th className="px-6 py-4 text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">Phone</th>
                      <th className="px-6 py-4 text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">Role</th>"""

code = code.replace(old_headers, new_headers)

# Update colSpan in empty state
code = code.replace('colSpan={6}', 'colSpan={8}')

# Update Table Body Cells
old_body = """                          <td className="px-6 py-4 text-[var(--text)] font-medium">
                            {account.username}
                          </td>
                          <td className="px-6 py-4">
                            <span className={`px-2.5 py-1 text-xs font-medium rounded-full ${"""

new_body = """                          <td className="px-6 py-4 text-[var(--text)] font-medium">
                            {account.username}
                          </td>
                          <td className="px-6 py-4 text-[var(--text-muted)] text-sm">
                            {account.email || '-'}
                          </td>
                          <td className="px-6 py-4 text-[var(--text-muted)] text-sm">
                            {account.phone || '-'}
                          </td>
                          <td className="px-6 py-4">
                            <span className={`px-2.5 py-1 text-xs font-medium rounded-full ${"""

code = code.replace(old_body, new_body)

with open('e:/aitrade/frontend/src/pages/AdminDashboard.tsx', 'w', encoding='utf-8') as f:
    f.write(code)

print("done")
